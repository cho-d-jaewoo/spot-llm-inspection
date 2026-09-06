"""SPOT 공장 점검 v2 메인 — Isaac Sim 순찰 + 2계층(Fixed/LLM-Full/LLM+HiTL) 의사결정 동시 수집."""

import argparse, os, random, sys
try:  # Windows cp949 콘솔에서 ━/이모지/한글 출력 깨짐 방지
    sys.stdout.reconfigure(encoding="utf-8"); sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass
import numpy as np

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--models", type=str, nargs="+", default=["haiku"],
                    help="LLM 별칭 1개 이상: haiku sonnet gpt4o llama")
parser.add_argument("--laps", type=int, default=10, help="총 순찰 lap 수")
parser.add_argument("--seed", type=int, default=42, help="anomaly 랜덤 생성 시드")
parser.add_argument("--output", type=str, default="results/metrics_raw.csv",
                    help="결과 CSV 경로")
parser.add_argument("--log-dir", type=str, default="results/logs",
                    help="모델별 실행 로그 디렉토리")
parser.add_argument("--headless", action="store_true", help="GUI 없이 실행")
parser.add_argument("--follow-cam", action="store_true", help="SPOT 추적 카메라")
parser.add_argument("--speed", type=int, default=2, choices=[1, 2, 3, 4],
                    help="시뮬레이션 속도 배율")
parser.add_argument("--record", action="store_true", help="viewport 영상 녹화")
parser.add_argument("--record-fps", type=int, default=20, help="녹화 FPS")
args = parser.parse_args()

simulation_app = SimulationApp({"headless": args.headless, "anti_aliasing": 0})

# ── Core imports (SimulationApp 이후) ─────────────────────────────────────────
from isaacsim.core.api import World
from isaacsim.core.api.objects import VisualCuboid, VisualCylinder
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.robot.policy.examples.robots.spot import SpotFlatTerrainPolicy

# ── v2 의사결정 모듈 (src/) ───────────────────────────────────────────────────
from anomaly_config import (ANOMALY_TYPES, ANOMALY_CONFIG, compute_severity, get_ground_truth,
                            get_fixed_decision, sample_attributes, build_prompt, is_ood)
from llm_decision import LLMDecisionClient
from oracle import oracle_tier2
from metrics_collector import MetricsCollector, DecisionRecord

# ═════════════════════════════════════════════════════════════════════════════
#  설정 (기존 코드 유지)
# ═════════════════════════════════════════════════════════════════════════════
ENV_USD = ("https://omniverse-content-production.s3-us-west-2.amazonaws.com"
           "/Assets/Isaac/5.0/Isaac/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd")
SPOT_USD = ("https://omniverse-content-production.s3-us-west-2.amazonaws.com"
            "/Assets/Isaac/5.0/Isaac/Robots/BostonDynamics/spot/spot.usd")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ICON_DIR = os.path.join(_ROOT, "assets", "icons")
ICON_FILES = {"FIRE_RISK": "icon_fire.png", "LIQUID_LEAK": "icon_leak.png",
              "AGV_STOPPED": "icon_agv.png", "SMOKE_DETECTED": "icon_smoke.png"}

POLICY_PATH = os.path.join(_ROOT, "policies", "spot_policy.pt")
ENV_YAML    = os.path.join(_ROOT, "policies", "spot_env.yaml")

PHYSICS_DT   = 0.002
RENDERING_DT = 0.002

ZONES = {
    "ZONE_A": {"name": "Assembly Line A", "center": (-7.0, -5.0)},
    "ZONE_B": {"name": "Storage Area B",  "center": (-2.5,  5.0)},
    "ZONE_C": {"name": "AGV Workspace",   "center": (-4.6, 15.6)},
    "ZONE_D": {"name": "Packaging Area",  "center": ( 7.0, 11.0)},
    "ZONE_E": {"name": "Loading Dock",    "center": ( 2.5, -1.8)},
}

PATROL_WAYPOINTS = [(-5.0, -8.0), (-5.0, 0.0), (-5.0, 8.0), (-5.0, 14.0),
                    (5.0, 14.0), (5.0, 8.0), (5.0, 0.0), (5.0, -8.0)]

# 제어 파라미터
WP_REACH_DIST  = 1.0
SPOT_SPEED     = 2.8
SPOT_TURN_GAIN = 3.0


# ═════════════════════════════════════════════════════════════════════════════
#  Anomaly 생성 (anomaly_config 사용)
# ═════════════════════════════════════════════════════════════════════════════
def severity_to_red(score: float) -> np.ndarray:
    """0~1 severity → 빨강 톤 그라데이션 (낮음=옅은 핑크, 높음=진한 빨강)."""
    score = max(0.0, min(1.0, score))
    return np.array([0.95 - 0.35 * score, 0.55 - 0.50 * score, 0.55 - 0.50 * score])


def generate_anomalies(seed=None):
    """ZONE마다 anomaly type 랜덤 배정 + 속성 랜덤 추출 + prompt/severity 생성."""
    rng = random.Random(seed)
    anomalies = []
    for zone_id, zone_info in ZONES.items():
        atype = rng.choice(ANOMALY_TYPES)
        attrs = sample_attributes(atype, rng)
        anomalies.append({
            "id": f"{atype}_{zone_id}", "type": atype, "zone": zone_id,
            "zone_name": zone_info["name"], "position": zone_info["center"],
            "radius": 2.5, "color": ANOMALY_CONFIG[atype]["color"], "attributes": attrs,
            "prompt": build_prompt(atype, zone_info["name"], attrs),
            "severity_score": compute_severity(atype, attrs),
        })
    return anomalies


# ═════════════════════════════════════════════════════════════════════════════
#  시나리오 시각화 (기존 코드 유지)
# ═════════════════════════════════════════════════════════════════════════════
def hide_overhead_prims(root_path="/World/Factory"):
    """천장/조명메쉬/기둥/보/벽 등 부감 시점 가리는 prim 숨김 (실제 광원은 보존)."""
    from pxr import Usd, UsdGeom
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        print(f"[ENV] hide_overhead_prims: {root_path} not found"); return
    keywords = ["ceiling", "roof", "lamp", "beam", "pillar", "truss",
                "rafter", "girder", "bracket", "slot", "window"]
    hidden = 0
    for prim in Usd.PrimRange(root):
        name = prim.GetName().lower()
        if "Light" in prim.GetTypeName():
            continue
        if any(kw in name for kw in keywords):
            try:
                UsdGeom.Imageable(prim).MakeInvisible(); hidden += 1
            except Exception:
                pass
    print(f"[ENV] Hidden {hidden} overhead/wall prims")


def deactivate_prims(root_path, keywords):
    """특정 키워드 prim 완전 비활성화 (visual + collision 모두 제거)."""
    from pxr import Usd
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        return
    deactivated = 0
    for prim in Usd.PrimRange(root):
        name = prim.GetName().lower()
        if any(kw in name for kw in keywords):
            prim.SetActive(False); deactivated += 1
    print(f"[ENV] Deactivated {deactivated} prims (keywords: {keywords})")


def spawn_factory_floor(x_range=(-50, 50), y_range=(-35, 35), tile_size=5.0, z=0.03):
    """공장 콘크리트 바닥 체크무늬 타일 (/World/FloorTiles 하위)."""
    from pxr import UsdGeom
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    UsdGeom.Xform.Define(stage, "/World/FloorTiles")
    tile_count = 0
    nx = int((x_range[1] - x_range[0]) / tile_size)
    ny = int((y_range[1] - y_range[0]) / tile_size)
    light_color = np.array([0.30, 0.30, 0.32]); dark_color = np.array([0.24, 0.24, 0.26])
    for i in range(nx):
        for j in range(ny):
            x = x_range[0] + (i + 0.5) * tile_size
            y = y_range[0] + (j + 0.5) * tile_size
            c = light_color if (i + j) % 2 == 0 else dark_color
            VisualCuboid(prim_path=f"/World/FloorTiles/Tile_{i}_{j}", name=f"tile_{i}_{j}",
                         position=np.array([x, y, z]),
                         scale=np.array([tile_size, tile_size, 0.01]), color=c)
            tile_count += 1
    print(f"[FLOOR] {tile_count} tiles ({nx}×{ny}, tile={tile_size}m)")


def create_textured_plane(prim_path, position, scale_xy, texture_path, rotation_axis=None):
    """텍스처 매핑된 양면 plane (anomaly type 아이콘 billboard)."""
    from pxr import UsdGeom, UsdShade, Sdf, Gf
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    sx, sy = scale_xy[0] / 2, scale_xy[1] / 2
    if rotation_axis is None:
        points = [Gf.Vec3f(-sx, -sy, 0), Gf.Vec3f(sx, -sy, 0), Gf.Vec3f(sx, sy, 0), Gf.Vec3f(-sx, sy, 0)]
    elif rotation_axis == 'x':
        points = [Gf.Vec3f(0, -sx, -sy), Gf.Vec3f(0, sx, -sy), Gf.Vec3f(0, sx, sy), Gf.Vec3f(0, -sx, sy)]
    else:  # 'y'
        points = [Gf.Vec3f(-sx, 0, -sy), Gf.Vec3f(sx, 0, -sy), Gf.Vec3f(sx, 0, sy), Gf.Vec3f(-sx, 0, sy)]
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateDoubleSidedAttr(True)
    uvs = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                                  interpolation=UsdGeom.Tokens.faceVarying)
    uvs.Set([Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(1, 1), Gf.Vec2f(0, 1)])
    UsdGeom.XformCommonAPI(mesh).SetTranslate(Gf.Vec3d(*position))
    mat_path = f"{prim_path}_Mat"
    material = UsdShade.Material.Define(stage, mat_path)
    surface = UsdShade.Shader.Define(stage, f"{mat_path}/Surface")
    surface.CreateIdAttr("UsdPreviewSurface")
    surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
    surface.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    texture = UsdShade.Shader.Define(stage, f"{mat_path}/Texture")
    texture.CreateIdAttr("UsdUVTexture")
    texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(texture_path))
    st_reader = UsdShade.Shader.Define(stage, f"{mat_path}/STReader")
    st_reader.CreateIdAttr("UsdPrimvarReader_float2")
    st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_reader.ConnectableAPI(), "result")
    surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(texture.ConnectableAPI(), "rgb")
    material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(material)


def despawn_anomaly_markers(anomalies):
    """기존 anomaly 시각 객체(zone/pole/icon) 모두 삭제 — 다음 lap 재스폰용."""
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    for a in anomalies:
        for prefix in ("AnomalyZone", "AnomalyPole", "AnomalyIcon"):
            path = f"/World/{prefix}_{a['id']}"
            if stage.GetPrimAtPath(path).IsValid():
                stage.RemovePrim(path)


def spawn_anomaly_markers(anomalies, world):
    """anomaly 시각화: 바닥 감지반경 원판(severity 색) + 수직 마커 + type 아이콘."""
    POLE_GRAY = np.array([0.35, 0.35, 0.38])
    for a in anomalies:
        x, y = a["position"]
        zone_color = severity_to_red(a["severity_score"])
        VisualCylinder(prim_path=f"/World/AnomalyZone_{a['id']}", name=f"zone_{a['id']}",
                       position=np.array([x, y, 0.05]), radius=a["radius"], height=0.02, color=zone_color)
        try:
            from pxr import UsdGeom
            import omni.usd
            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(f"/World/AnomalyZone_{a['id']}")
            if prim.IsValid():
                UsdGeom.Gprim(prim).CreateDisplayOpacityAttr().Set([1.0])
        except Exception:
            pass
        VisualCuboid(prim_path=f"/World/AnomalyPole_{a['id']}", name=f"pole_{a['id']}",
                     position=np.array([x, y, 1.1]), scale=np.array([0.06, 0.06, 2.2]), color=POLE_GRAY)
        icon_file = ICON_FILES.get(a["type"])
        tex_path = os.path.join(ICON_DIR, icon_file) if icon_file else None
        if tex_path and os.path.exists(tex_path):
            create_textured_plane(prim_path=f"/World/AnomalyIcon_{a['id']}", position=(x, y, 2.55),
                                  scale_xy=(1.0, 1.0), texture_path=tex_path.replace("\\", "/"), rotation_axis='y')
        else:
            if tex_path:
                print(f"[WARN] icon file not found: {tex_path}")
            VisualCuboid(prim_path=f"/World/AnomalyIcon_{a['id']}", name=f"icon_{a['id']}",
                         position=np.array([x, y, 2.45]), scale=np.array([0.5, 0.5, 0.5]),
                         color=np.array(a["color"]))
        print(f"[ANOMALY] {a['zone']:7s} → {a['type']:15s} at ({x:5.1f},{y:5.1f})")


def spawn_waypoint_markers():
    """순찰 waypoint를 작은 디스크로 생성 (logic 유지, 시각만 invisible)."""
    from pxr import UsdGeom
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    for i, (x, y) in enumerate(PATROL_WAYPOINTS):
        VisualCuboid(prim_path=f"/World/Waypoint_{i}", name=f"wp_{i}",
                     position=np.array([x, y, 0.06]), scale=np.array([2.0, 2.0, 0.05]),
                     color=np.array([0.0, 0.85, 0.65]))
        prim = stage.GetPrimAtPath(f"/World/Waypoint_{i}")
        if prim.IsValid():
            UsdGeom.Imageable(prim).MakeInvisible()


# ═════════════════════════════════════════════════════════════════════════════
#  Action Executor (기존 코드 그대로 재사용 — Tier1 물리 실행)
# ═════════════════════════════════════════════════════════════════════════════
class ActionExecutor:
    """Tier1 action(APPROACH/WAIT_AND_OBSERVE/KEEP_DISTANCE)을 sub-state machine으로 수행.
       INIT → POSITIONING → ARCING → WAITING → DONE. radius/arc 범위만 action별 차등."""

    ACTION_PARAMS = {
        "APPROACH":         {"radius": 1.5, "arc_neg_deg": -45, "arc_pos_deg": 90},
        "WAIT_AND_OBSERVE": {"radius": 2.5, "arc_neg_deg": -30, "arc_pos_deg": 60},
        "KEEP_DISTANCE":    {"radius": 4.0, "arc_neg_deg": -60, "arc_pos_deg": 120},
    }
    DIST_TOLERANCE = 0.3
    ARC_SPEED      = 1.2
    POST_WAIT_TIME = 3.0

    def __init__(self, action, anomaly, spot_ctrl):
        self.action = action; self.anomaly = anomaly; self.spot_ctrl = spot_ctrl
        self.phase = "INIT"; self.arc_angle = 0.0

    def is_done(self):
        return self.phase == "DONE"

    def update(self, dt):
        if self.action not in self.ACTION_PARAMS:
            self.phase = "DONE"; return np.array([0.0, 0.0, 0.0])
        return self._arc_observe(dt)

    def _arc_observe(self, dt):
        params = self.ACTION_PARAMS[self.action]
        radius = params["radius"]
        arc_neg = np.deg2rad(params["arc_neg_deg"]); arc_pos = np.deg2rad(params["arc_pos_deg"])
        pos, yaw = self.spot_ctrl.get_pose()
        ap = np.array(self.anomaly["position"]); diff = ap - pos
        dist = float(np.linalg.norm(diff))

        if self.phase == "INIT":
            print(f"  [ACTION] {self.action} → radius {radius}m "
                  f"(arc {params['arc_neg_deg']}° ~ +{params['arc_pos_deg']}°)")
            self.phase = "POSITIONING"

        if self.phase == "POSITIONING":
            tgt_yaw = float(np.arctan2(diff[1], diff[0]))
            yaw_err = (tgt_yaw - yaw + np.pi) % (2 * np.pi) - np.pi
            wz = float(np.clip(3.0 * yaw_err, -2.0, 2.0))
            if abs(yaw_err) > 0.4:
                return np.array([0.0, 0.0, wz])
            if dist < radius - self.DIST_TOLERANCE:
                return np.array([-1.5, 0.0, wz])
            if dist > radius + self.DIST_TOLERANCE:
                return np.array([1.5, 0.0, wz])
            self.arc_start = float(np.arctan2(-diff[1], -diff[0]))
            self.arc_angle = self.arc_start
            self.arc_target_neg = self.arc_start + arc_neg
            self.arc_target_pos = self.arc_start + arc_pos
            self.arc_sub = "TO_NEG"
            print(f"  [ACTION] {self.action} → 거리 {dist:.1f}m 확보, oscillation 시작")
            self.phase = "ARCING"
            return np.array([0.0, 0.0, 0.0])

        if self.phase == "ARCING":
            if self.arc_sub == "TO_NEG":
                self.arc_angle -= dt * self.ARC_SPEED
                if self.arc_angle <= self.arc_target_neg:
                    self.arc_sub = "TO_POS"
            elif self.arc_sub == "TO_POS":
                self.arc_angle += dt * self.ARC_SPEED
                if self.arc_angle >= self.arc_target_pos:
                    print(f"  [ACTION] {self.action} → arc_pos 도달, 대처 대기")
                    self.phase = "WAITING"; self.wait_timer = 0.0
                    return np.array([0.0, 0.0, 0.0])
            next_pt = np.array([ap[0] + radius * np.cos(self.arc_angle),
                                ap[1] + radius * np.sin(self.arc_angle)])
            dist_err = radius - dist
            radial_dir = -diff / (dist + 1e-6)
            move_dir = (next_pt - pos) + radial_dir * dist_err * 1.5
            move_norm = float(np.linalg.norm(move_dir))
            tgt_yaw = float(np.arctan2(diff[1], diff[0]))
            yaw_err = (tgt_yaw - yaw + np.pi) % (2 * np.pi) - np.pi
            wz = float(np.clip(3.0 * yaw_err, -2.0, 2.0))
            if move_norm < 1e-3:
                return np.array([0.0, 0.0, wz])
            move_dir /= move_norm
            spot_forward = np.array([np.cos(yaw), np.sin(yaw)])
            spot_left = np.array([-np.sin(yaw), np.cos(yaw)])
            vx = float(np.clip(np.dot(move_dir, spot_forward) * 2.0, -1.5, 1.5))
            vy = float(np.clip(np.dot(move_dir, spot_left) * 2.5, -1.0, 1.0))
            return np.array([vx, vy, wz])

        if self.phase == "WAITING":
            tgt_yaw = float(np.arctan2(diff[1], diff[0]))
            yaw_err = (tgt_yaw - yaw + np.pi) % (2 * np.pi) - np.pi
            wz = float(np.clip(3.0 * yaw_err, -2.0, 2.0))
            self.wait_timer += dt
            if self.wait_timer >= self.POST_WAIT_TIME:
                print(f"  [ACTION] {self.action} 완료")
                self.phase = "DONE"; return np.array([0.0, 0.0, 0.0])
            return np.array([0.0, 0.0, wz])

        return np.array([0.0, 0.0, 0.0])


# ═════════════════════════════════════════════════════════════════════════════
#  결정 처리 (3개 mode × 다중 모델 동시 수집)
# ═════════════════════════════════════════════════════════════════════════════
def process_anomaly_event(anomaly, lap, clients, collector, log_files):
    """anomaly 1건 처리 — Fixed/LLM-Full/LLM+HiTL × 모든 모델 기록.
       반환: 물리적으로 실행할 Tier1 action (첫 모델의 HiTL tier1 기준)."""
    atype = anomaly["type"]; attrs = anomaly["attributes"]
    gt = get_ground_truth(atype, attrs)
    severity_score = anomaly["severity_score"]
    fixed = get_fixed_decision(atype, attrs)

    print("\n" + "━" * 68)
    print(f"  🚨  ANOMALY DETECTED  —  {anomaly['id']}  (lap {lap})")
    print(f"      Zone       : {anomaly['zone']} ({anomaly['zone_name']})")
    print(f"      Type       : {atype}")
    print(f"      Attributes : {attrs}")
    print(f"      Severity   : {severity_score:.2f}")
    print(f"      [GT] tier1={gt['tier1']}  tier2={gt['tier2']}  escalate={gt['escalation']}")
    print(f"           ref: {gt['reference']}")
    print(f"      [FIXED] tier1={fixed['tier1']}  tier2={fixed['tier2']}")

    drive_tier1 = fixed["tier1"]   # LLM 실패 시 fallback
    for model, client in clients.items():
        try:
            llm, elapsed_ms = client.decide(anomaly["prompt"])
        except Exception as e:
            print(f"  [LLM:{model}] ERROR → fixed로 대체: {e}")
            llm = {"tier1_action": fixed["tier1"], "tier2_escalate": False,
                   "tier2_action": fixed["tier2"], "tier2_confidence": 0.5,
                   "severity": "MEDIUM", "reasoning": f"LLM error: {e}"}
            elapsed_ms = 0.0

        llm_full_tier2 = llm["tier2_action"]
        if llm["tier2_escalate"]:
            hitl_tier2_final = oracle_tier2(atype, attrs)
        else:
            hitl_tier2_final = llm["tier2_action"]

        record = DecisionRecord(
            lap=lap, zone_id=anomaly["zone"], anomaly_type=atype, is_ood=is_ood(atype),
            attributes=dict(attrs), severity_score=severity_score,
            tier1_gt=gt["tier1"], tier2_gt=gt["tier2"], escalation_gt=gt["escalation"],
            fixed_tier1=fixed["tier1"], fixed_tier2=fixed["tier2"],
            fixed_tier1_correct=(fixed["tier1"] == gt["tier1"]),
            fixed_tier2_correct=(fixed["tier2"] == gt["tier2"]),
            model=model,
            llm_full_tier1=llm["tier1_action"], llm_full_tier2=llm_full_tier2,
            llm_full_tier1_correct=(llm["tier1_action"] == gt["tier1"]),
            llm_full_tier2_correct=(llm_full_tier2 == gt["tier2"]),
            llm_full_decision_time_ms=round(elapsed_ms, 1),
            hitl_tier1=llm["tier1_action"], hitl_escalated=llm["tier2_escalate"],
            tier2_confidence=llm["tier2_confidence"], hitl_tier2_final=hitl_tier2_final,
            hitl_tier1_correct=(llm["tier1_action"] == gt["tier1"]),
            hitl_escalation_correct=(llm["tier2_escalate"] == gt["escalation"]),
            hitl_tier2_correct=(hitl_tier2_final == gt["tier2"]),
            hitl_decision_time_ms=round(elapsed_ms, 1),
            hitl_reasoning=llm["reasoning"],
        )
        collector.add(record)

        esc_mark = "→ORACLE" if llm["tier2_escalate"] else "→LLM"
        print(f"  [LLM:{model:7s}] ({elapsed_ms:5.0f}ms) tier1={llm['tier1_action']:16s} "
              f"esc={str(llm['tier2_escalate']):5s} conf={llm['tier2_confidence']:.2f} "
              f"tier2_full={llm_full_tier2:11s} tier2_hitl={hitl_tier2_final:11s} {esc_mark}")
        if model in log_files:
            f = log_files[model]
            f.write(f"[lap {lap}] {anomaly['id']} | attrs={attrs}\n")
            f.write(f"  GT: tier1={gt['tier1']} tier2={gt['tier2']} esc={gt['escalation']}\n")
            f.write(f"  LLM: tier1={llm['tier1_action']} esc={llm['tier2_escalate']} "
                    f"tier2_full={llm_full_tier2} tier2_hitl={hitl_tier2_final}\n")
            f.write(f"  reasoning: {llm['reasoning']}\n\n")
            f.flush()

        # 첫 모델이 물리 실행 Tier1 결정
        if model == next(iter(clients)):
            drive_tier1 = llm["tier1_action"]

    print("━" * 68)
    return drive_tier1


# ═════════════════════════════════════════════════════════════════════════════
#  SPOT 순찰 컨트롤러 (기존 구조 유지 — 의사결정만 v2로 교체)
# ═════════════════════════════════════════════════════════════════════════════
class SpotPatrolController:
    """PATROLLING(waypoint 추종 + anomaly 감지) → EXECUTING(Tier1 행동) → PATROLLING."""

    def __init__(self, spot, anomalies, decision_handler):
        self.spot = spot; self.anomalies = anomalies
        self.decision_handler = decision_handler   # (anomaly, lap) → tier1_action
        self.wp_idx = 0; self.current_lap = 1
        self.state = "PATROLLING"; self.current = None; self.executor = None
        self.processed = set()

    def get_pose(self):
        pos, q = self.spot.robot.get_world_pose()
        w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        yaw = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
        return np.array([float(pos[0]), float(pos[1])]), yaw

    def reset_for_new_lap(self, new_anomalies):
        self.anomalies = new_anomalies; self.processed = set()
        self.current = None; self.executor = None; self.state = "PATROLLING"

    def compute_command(self, target_xy):
        pos, yaw = self.get_pose()
        diff = np.array(target_xy) - pos
        dist = float(np.linalg.norm(diff))
        if dist < WP_REACH_DIST:
            return np.array([0.0, 0.0, 0.0]), True
        tgt_yaw = float(np.arctan2(diff[1], diff[0]))
        err = (tgt_yaw - yaw + np.pi) % (2 * np.pi) - np.pi
        wz = float(np.clip(SPOT_TURN_GAIN * err, -2.0, 2.0))
        vx = float(SPOT_SPEED * max(0.0, 1.0 - abs(err) / 1.2))
        return np.array([vx, 0.0, wz]), False

    def detect_anomaly(self):
        if self.state != "PATROLLING":
            return None
        pos, _ = self.get_pose()
        for a in self.anomalies:
            if a["id"] in self.processed:
                continue
            d = float(np.linalg.norm(pos - np.array(a["position"])))
            if d < a["radius"]:
                return a
        return None

    def on_detect(self, anomaly):
        tier1 = self.decision_handler(anomaly, self.current_lap)
        self.current = anomaly
        self.executor = ActionExecutor(tier1, anomaly, self)
        self.state = "EXECUTING"

    def on_execution_done(self):
        a = self.current
        print(f"[SPOT] Action 완료: {a['id']} → patrol 재개\n")
        self._mark_zone_processed(a)
        self.processed.add(a["id"])
        self.current = None; self.executor = None; self.state = "PATROLLING"

    def _mark_zone_processed(self, anomaly):
        try:
            from pxr import UsdGeom, Gf
            import omni.usd
            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(f"/World/AnomalyZone_{anomaly['id']}")
            if prim.IsValid():
                UsdGeom.Gprim(prim).GetDisplayColorAttr().Set([Gf.Vec3f(0.0, 0.85, 0.65)])
        except Exception as e:
            print(f"[WARN] zone 색상 변경 실패: {e}")

    def update(self, dt):
        if self.state == "PATROLLING":
            detected = self.detect_anomaly()
            if detected is not None:
                self.on_detect(detected)
                return np.array([0.0, 0.0, 0.0])
            target = PATROL_WAYPOINTS[self.wp_idx]
            cmd, reached = self.compute_command(target)
            if reached:
                self.wp_idx = (self.wp_idx + 1) % len(PATROL_WAYPOINTS)
            return cmd
        elif self.state == "EXECUTING":
            cmd = self.executor.update(dt)
            if self.executor.is_done():
                self.on_execution_done()
            return cmd
        return np.array([0.0, 0.0, 0.0])


# ═════════════════════════════════════════════════════════════════════════════
#  Viewport 영상 녹화 (기존 코드 유지)
# ═════════════════════════════════════════════════════════════════════════════
class FrameRecorder:
    """Isaac Sim viewport를 PNG 시퀀스로 캡처 → ffmpeg로 MP4 합성."""

    def __init__(self, base_dir="./recordings", fps=20):
        import datetime
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(base_dir, ts)
        self.frames_dir = os.path.join(self.session_dir, "frames")
        os.makedirs(self.frames_dir, exist_ok=True)
        self.fps = fps; self.frame_idx = 0
        self._capture_fn = None; self._viewport = None
        try:
            from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
            self._viewport = get_active_viewport()
            self._capture_fn = capture_viewport_to_file
            print(f"[REC] Recording enabled → {self.session_dir}  ({fps} FPS)")
        except Exception as e:
            print(f"[REC] WARNING: viewport API 불가 ({e}) — 녹화 비활성")

    def capture(self):
        if self._capture_fn is None:
            return
        path = os.path.join(self.frames_dir, f"frame_{self.frame_idx:06d}.png")
        try:
            self._capture_fn(self._viewport, file_path=path); self.frame_idx += 1
        except Exception as e:
            if self.frame_idx == 0:
                print(f"[REC] WARNING: capture failed ({e})")

    def finalize(self):
        if self.frame_idx == 0:
            print("[REC] No frames captured."); return
        out = os.path.join(self.session_dir, "simulation.mp4")
        pattern = os.path.join(self.frames_dir, "frame_%06d.png")
        import subprocess, shutil
        try:
            subprocess.run(["ffmpeg", "-y", "-framerate", str(self.fps), "-i", pattern,
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-loglevel", "error", out], check=True)
            print(f"[REC] Video saved → {out}  ({self.frame_idx} frames)")
            shutil.rmtree(self.frames_dir, ignore_errors=True)
        except FileNotFoundError:
            print(f"[REC] ffmpeg not found. PNG preserved: {self.frames_dir}")
        except subprocess.CalledProcessError as e:
            print(f"[REC] ffmpeg error: {e}")


# ═════════════════════════════════════════════════════════════════════════════
#  메인
# ═════════════════════════════════════════════════════════════════════════════
def main():
    # ── LLM 클라이언트 초기화 (다중 모델) ────────────────────────────────────
    clients = {}
    for m in args.models:
        clients[m] = LLMDecisionClient(m)
    collector = MetricsCollector()

    # ── 모델별 로그 파일 ──────────────────────────────────────────────────────
    os.makedirs(args.log_dir, exist_ok=True)
    log_files = {m: open(os.path.join(args.log_dir, f"log_{m}.txt"), "w", encoding="utf-8")
                 for m in args.models}

    world = World(stage_units_in_meters=1.0, physics_dt=PHYSICS_DT, rendering_dt=RENDERING_DT)

    # ── 공장 환경 ─────────────────────────────────────────────────────────────
    world.scene.add_default_ground_plane()
    add_reference_to_stage(usd_path=ENV_USD, prim_path="/World/Factory")
    print("[ENV] warehouse_multiple_shelves loaded")
    hide_overhead_prims("/World/Factory")
    deactivate_prims("/World/Factory", ["palletbin"])
    spawn_factory_floor()

    from isaacsim.core.utils.viewports import set_camera_view
    set_camera_view(eye=np.array([0.0, -22.0, 30.0]), target=np.array([0.0, 4.0, 0.0]))

    anomalies = generate_anomalies(seed=args.seed)
    spawn_waypoint_markers()
    spawn_anomaly_markers(anomalies, world)

    # ── SPOT ──────────────────────────────────────────────────────────────────
    spot = SpotFlatTerrainPolicy(prim_path="/World/SPOT", name="spot", usd_path=SPOT_USD,
                                 position=np.array([0.0, -7.0, 0.75]))
    spot.load_policy(POLICY_PATH, ENV_YAML)
    world.reset()
    spot.initialize(); spot.post_reset()
    spot.robot.set_joint_positions(spot.default_pos)
    spot.robot.set_joint_velocities(spot.default_vel)
    for _ in range(200):
        world.step(render=False)

    decision_handler = lambda anomaly, lap: process_anomaly_event(
        anomaly, lap, clients, collector, log_files)
    patrol = SpotPatrolController(spot, anomalies, decision_handler)

    print("\n" + "=" * 68)
    print("  SPOT Factory Inspection v2  —  2-Tier LLM-Human Collaboration")
    print(f"  Models       : {', '.join(args.models)}")
    print(f"  Zones        : {len(ZONES)}   Anomalies/lap : {len(anomalies)} (seed={args.seed})")
    print(f"  Waypoints    : {len(PATROL_WAYPOINTS)}   Total laps : {args.laps}")
    print(f"  Modes        : Fixed | LLM-Full | LLM+HiTL (동시 수집)")
    print(f"  Output       : {args.output}")
    print("=" * 68 + "\n")

    if args.follow_cam:
        from isaacsim.core.utils.viewports import set_camera_view as _set_cam

    recorder = FrameRecorder(base_dir="./recordings", fps=args.record_fps) if args.record else None

    step = 0
    all_processed_announced = False; lap_completed_announced = False; home_return_announced = False
    prev_wp_idx = patrol.wp_idx; visited_wps = set()
    SPOT_HOME = np.array([0.0, -7.0]); HOME_RADIUS = 1.5

    print(f"\n[SIM] Starting Lap 1 / {args.laps}\n")

    while simulation_app.is_running():
        for sub in range(args.speed):
            cmd = patrol.update(PHYSICS_DT)
            spot.forward(dt=PHYSICS_DT, command=cmd)
            world.step(render=(sub == args.speed - 1))
            step += 1

        if recorder is not None:
            recorder.capture()

        if patrol.state == "PATROLLING":
            visited_wps.add(patrol.wp_idx)
            if (patrol.wp_idx != prev_wp_idx and patrol.wp_idx == 0
                    and len(visited_wps) >= len(PATROL_WAYPOINTS)):
                if not lap_completed_announced:
                    print(f"\n[SIM] Lap {patrol.current_lap} waypoints complete. Returning to start...\n")
                    lap_completed_announced = True
        prev_wp_idx = patrol.wp_idx

        if args.follow_cam:
            pos, yaw = patrol.get_pose()
            backward = np.array([np.cos(yaw + np.pi), np.sin(yaw + np.pi)])
            eye = (float(pos[0] + backward[0] * 4.0), float(pos[1] + backward[1] * 7.0), 8.0)
            _set_cam(eye=np.array(eye), target=np.array([float(pos[0]), float(pos[1]), 2]))

        if (len(patrol.processed) == len(anomalies) and patrol.state == "PATROLLING"
                and not all_processed_announced):
            print(f"\n[SIM] Lap {patrol.current_lap}: all {len(anomalies)} anomalies processed.\n")
            all_processed_announced = True

        if lap_completed_announced and patrol.state == "PATROLLING":
            pos, _ = patrol.get_pose()
            if float(np.linalg.norm(pos - SPOT_HOME)) < HOME_RADIUS:
                if not home_return_announced:
                    print(f"\n[SIM] SPOT returned home. Lap {patrol.current_lap} complete.\n")
                    home_return_announced = True
                if patrol.current_lap >= args.laps:
                    print(f"[SIM] All {args.laps} laps completed.\n")
                    break
                else:
                    next_lap = patrol.current_lap + 1
                    print(f"[SIM] Re-spawning anomalies for Lap {next_lap}...")
                    despawn_anomaly_markers(anomalies)
                    new_seed = args.seed + next_lap if args.seed is not None else None
                    anomalies = generate_anomalies(seed=new_seed)
                    spawn_anomaly_markers(anomalies, world)
                    patrol.current_lap = next_lap
                    patrol.reset_for_new_lap(anomalies)
                    visited_wps = set()
                    lap_completed_announced = False; home_return_announced = False
                    all_processed_announced = False
                    print(f"\n[SIM] Starting Lap {next_lap} / {args.laps}\n")

    # ── 결과 저장 + 요약 ──────────────────────────────────────────────────────
    collector.save_csv(args.output)
    collector.print_summary()
    for f in log_files.values():
        f.close()
    if recorder is not None:
        recorder.finalize()


if __name__ == "__main__":
    main()
    simulation_app.close()
