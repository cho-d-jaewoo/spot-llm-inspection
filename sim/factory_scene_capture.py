"""
factory_scene_capture.py  —  논문용 환경 캡처 전용 (정적 씬)
============================================================
Isaac Sim 5.0  |  물리/이동 없음. 씬을 열어두고 사용자가 직접 시점 조정 후 수동 캡처.

구성:
  - NVIDIA warehouse_multiple_shelves USD 로드 (선반/기둥/팔레트/벽 등 실제 구조물)
    → spot_factory_inspection.py와 동일 환경. 천장/조명메쉬만 숨겨 부감 가능.
  - 정사각형 콘크리트 타일 바닥 (40 × 40 m, 5 m 타일 → 8×8 = 64 tiles)
  - Clearpath Jackal AGV × 4대 (바디 빨강, 등판 위 SO-101 Manipulator 회색)
  - DomeLight + 상단 DiskLight 조명

실행:
  python factory_scene_capture.py             # GUI (창 유지, 수동 캡처)
  python factory_scene_capture.py --headless  # 창 없이 (문법 점검용)
  python factory_scene_capture.py --no-warehouse  # warehouse 없이 빈 바닥만

동작:
  Launch → build scene → 렌더링 갱신(물리 없음) → 창 유지
  자동 캡처 / 자동 종료 / 물리 루프 없음.

참고:
  - warehouse 구조물 좌표 확인: 콘솔의 [INSPECT] 출력 참고
  - 특정 구조물 숨기려면 hide_overhead_prims의 keywords 수정
  - 선반/박스 복제·크기조절은 GUI(Stage 트리)에서 직접

USD 경로 (Isaac Sim 5.0):
  get_assets_root_path()로 에셋 루트 자동 탐색 (로컬/S3 자동 적응).
  - Jackal   : <root>/Isaac/Robots/Clearpath/Jackal/jackal_basic.usd
  - SO-101   : <root>/Isaac/Robots/RobotStudio/so101_new_calib/so101_new_calib.usd
  - warehouse   : <root>/Isaac/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd
  ※ 로드 실패 시 콘솔에 [!] 경고 출력 (조용히 넘어가지 않음).
     경로가 틀리면 해당 USD 상수 수정. 색상은 FLOOR_COLOR/JACKAL_COLOR/ARM_COLOR로 조정.
"""

# ── Isaac Sim 앱 초기화 (반드시 최상단) ─────────────────────────────────────
from isaacsim import SimulationApp
import argparse

_ap = argparse.ArgumentParser(description="Factory 정적 씬 캡처용 뷰어")
_ap.add_argument("--headless", action="store_true", help="GUI 없이 실행 (문법 점검용)")
_ap.add_argument("--no-warehouse", action="store_true",
                 help="warehouse USD 없이 빈 바닥 + AGV만 (구조물 제외)")
_args, _ = _ap.parse_known_args()

simulation_app = SimulationApp({"headless": _args.headless,
                                "renderer": "RayTracedLighting"})

# ── 일반 import (SimulationApp 인스턴스화 이후) ──────────────────────────────
import numpy as np
import omni.usd
from pxr import Usd, UsdGeom, UsdLux, UsdPhysics, Sdf, Gf

from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path

# ── 에셋 루트 경로 (Isaac Sim 5.0 공식 헬퍼) ──────────────────────────────────
# get_assets_root_path()는 로컬 에셋 또는 S3를 자동 탐색.
# 하드코딩 S3 URL보다 안전 (사용자 환경에 맞춰 자동 적응).
ASSETS_ROOT = get_assets_root_path()
if ASSETS_ROOT is None:
    raise RuntimeError(
        "Isaac Sim 에셋 루트를 찾을 수 없음. 인터넷 연결 또는 로컬 에셋 설치 확인."
    )
print(f"[ASSETS] root = {ASSETS_ROOT}")

# ── USD 경로 (Isaac Sim 5.0) ─────────────────────────────────────────────────
# 공장 환경 — spot_factory_inspection.py와 동일 (선반/기둥/팔레트/벽 포함)
ENV_USD = f"{ASSETS_ROOT}/Isaac/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd"

# Clearpath Jackal — /Isaac/Robots/Clearpath/Jackal/jackal_basic.usd
#   jackal_basic.usd : 기본 버전 (차체+바퀴, 센서 없음 → 등판 깔끔, 팔 얹기 좋음)
#   jackal.usd       : 풀 버전 (센서 포함)
JACKAL_USD = f"{ASSETS_ROOT}/Isaac/Robots/Clearpath/Jackal/jackal_basic.usd"

# 로봇팔 — SO-101 (RobotStudio) manipulator
ARM_USD = f"{ASSETS_ROOT}/Isaac/Robots/RobotStudio/so101_new_calib/so101_new_calib.usd"

# ── 색상 설정 (RGB 0~1) ──────────────────────────────────────────────────────
FLOOR_COLOR  = (0.004, 0.18, 0.075)   # 매우 진한 팩토리 녹색 (바닥 타일)
JACKAL_COLOR = (0.231, 0.325, 0.522)   #
ARM_COLOR    = (0.58, 0.549, 0.549)   # 회색 (manipulator)

# ── Jackal 등판 위 manipulator mount 높이 (m) ────────────────────────────────
# Jackal 차체 상단 ≈ 0.25 m. 부정확하면 GUI에서 직접 조정.
JACKAL_DECK_Z = 0.25

# ── AGV 배치 (4대, 40×40 환경에서 충분히 떨어지게) ───────────────────────────
# 4분면에 하나씩, 중심에서 ±12 m → 서로 24 m 간격 (절대 안 겹침)
AGV_CONFIGS = [
    {"name": "Jackal_A", "xy": (-12.0, -12.0), "yaw_deg":  45.0},  # 좌하
    {"name": "Jackal_B", "xy": ( 12.0, -12.0), "yaw_deg": 135.0},  # 우하
    {"name": "Jackal_C", "xy": (-12.0,  12.0), "yaw_deg": 315.0},  # 좌상
    {"name": "Jackal_D", "xy": ( 12.0,  12.0), "yaw_deg": 225.0},  # 우상
]


# ═════════════════════════════════════════════════════════════════════════════
#  유틸
# ═════════════════════════════════════════════════════════════════════════════

def _set_color(prim, rgb: tuple):
    """USD prim에 displayColor 설정 (보조용)."""
    gprim = UsdGeom.Gprim(prim)
    attr = gprim.GetDisplayColorAttr()
    if not attr:
        attr = gprim.CreateDisplayColorAttr()
    attr.Set([Gf.Vec3f(*rgb)])


# 색상별 머티리얼 캐시 (같은 색은 머티리얼 1개 재사용)
_MAT_CACHE = {}

def _get_or_create_material(rgb: tuple):
    """주어진 색의 OmniPBR(UsdPreviewSurface) 머티리얼을 만들어 반환.
       displayColor는 기존 머티리얼에 가려지므로, 실제 머티리얼을 새로 바인딩해야 색이 바뀜.
    """
    from pxr import UsdShade
    key = tuple(round(c, 3) for c in rgb)
    if key in _MAT_CACHE:
        return _MAT_CACHE[key]

    stage = omni.usd.get_context().get_stage()
    UsdGeom.Scope.Define(stage, "/World/Looks")   # 머티리얼 보관 scope
    mat_name = "Mat_" + "_".join(str(int(c * 255)) for c in rgb)
    mat_path = f"/World/Looks/{mat_name}"

    mat = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, f"{mat_path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    _MAT_CACHE[key] = mat
    return mat


def colorize_subtree(root_path: str, rgb: tuple, skip_keywords=None,
                     exclude_paths=None):
    """root_path 하위의 모든 Mesh에 새 머티리얼을 바인딩해서 색 변경.

       [중요] displayColor는 기존 머티리얼(MDL)에 가려져서 화면에 안 보임.
       → 새 UsdPreviewSurface 머티리얼을 만들어 BindMaterial로 강제 적용해야
         실제로 색이 바뀜.

       skip_keywords: 이 단어가 prim 이름에 있으면 색 안 바꿈
       exclude_paths: 이 경로(및 하위)는 색칠 제외
    """
    from pxr import UsdShade
    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        return 0
    skip_keywords = skip_keywords or []
    exclude_paths = exclude_paths or []

    mat = _get_or_create_material(rgb)
    painted = 0
    for prim in Usd.PrimRange(root):
        # Mesh 타입만 (실제 렌더링되는 지오메트리)
        if prim.GetTypeName() != "Mesh":
            continue
        prim_path_str = str(prim.GetPath())
        if any(prim_path_str.startswith(ex) for ex in exclude_paths):
            continue
        name_lc = prim.GetName().lower()
        if any(kw in name_lc for kw in skip_keywords):
            continue
        try:
            # 머티리얼 바인딩 (기존 머티리얼 덮어씀, strength=strongerThanDescendants)
            UsdShade.MaterialBindingAPI(prim).Bind(
                mat, bindingStrength=UsdShade.Tokens.strongerThanDescendants)
            _set_color(prim, rgb)   # displayColor도 같이 (fallback)
            painted += 1
        except Exception:
            pass
    return painted


# ═════════════════════════════════════════════════════════════════════════════
#  씬 구성 함수 (씬 요소당 함수 1개)
# ═════════════════════════════════════════════════════════════════════════════

def spawn_square_floor(half: float = 20.0, tile_size: float = 5.0, z: float = 0.0):
    """정사각형 콘크리트 타일 바닥.
       half=20 → 40 m × 40 m, tile_size=5 → 8×8 = 64 tiles, 두께 0.02 m.
    """
    stage = omni.usd.get_context().get_stage()
    UsdGeom.Xform.Define(stage, "/World/FloorTiles")
    from pxr import UsdShade
    floor_mat = _get_or_create_material(FLOOR_COLOR)
    n = int(2 * half / tile_size)
    count = 0
    for i in range(n):
        for j in range(n):
            x = -half + (i + 0.5) * tile_size
            y = -half + (j + 0.5) * tile_size
            path = f"/World/FloorTiles/Tile_{i}_{j}"
            cube = UsdGeom.Cube.Define(stage, path)
            cube.GetSizeAttr().Set(1.0)
            xform = UsdGeom.XformCommonAPI(cube)
            xform.SetTranslate((x, y, z))
            xform.SetScale((tile_size, tile_size, 0.02))  # 매우 얇은 타일
            _set_color(cube.GetPrim(), FLOOR_COLOR)
            UsdShade.MaterialBindingAPI(cube.GetPrim()).Bind(
                floor_mat, bindingStrength=UsdShade.Tokens.strongerThanDescendants)
            count += 1
    print(f"[FLOOR] {count} tiles ({n}×{n}), {2*half}m × {2*half}m")


def _add_ref_checked(usd_path: str, prim_path: str, label: str) -> bool:
    """add_reference_to_stage + 로드 검증.
       reference 추가 후 prim에 실제 children이 생겼는지 확인.
       404 등으로 빈 prim만 생기면 False 반환 + 경고 출력.
    """
    stage = omni.usd.get_context().get_stage()
    add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        print(f"  [!] {label}: prim 생성 실패 → {usd_path}")
        return False
    # reference가 제대로 로드되면 children(메쉬/joint 등)이 생김
    children = list(prim.GetChildren())
    if len(children) == 0:
        print(f"  [!] {label}: reference 로드됐으나 비어있음 (경로 404 의심)")
        print(f"      USD: {usd_path}")
        return False
    return True


def make_static(root_path: str):
    """root_path 하위의 모든 RigidBody 물리를 비활성화 → 고정(static).
       그림 캡처용이라 물리 시뮬 불필요. 툭 쳐도 안 날아가게.

       방법: 모든 prim에서
         - UsdPhysics.RigidBodyAPI 의 physics:rigidBodyEnabled = False
         - articulation root 의 physics:articulationEnabled = False
       이렇게 하면 중력/충돌 반응 없이 제자리에 고정됨.
    """
    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        return 0
    disabled = 0
    for prim in Usd.PrimRange(root):
        # RigidBody 비활성화
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rb = UsdPhysics.RigidBodyAPI(prim)
            rb.GetRigidBodyEnabledAttr().Set(False)
            disabled += 1
        # Articulation root 비활성화 (로봇 관절 체인)
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            art = UsdPhysics.ArticulationRootAPI(prim)
            try:
                art.GetArticulationEnabledAttr().Set(False)
            except Exception:
                pass
    return disabled


def add_robot_at(usd_path: str, wrapper_path: str, label: str,
                 xy: tuple, z: float, yaw_deg: float) -> bool:
    """로봇 USD를 wrapper Xform 안에 넣고 위치/회전 적용.

    [왜 wrapper를 쓰나]
      로봇 USD 최상위 prim은 보통 자체 xformOp를 갖고 있어서
      거기에 XformCommonAPI.SetTranslate를 직접 쓰면
      "incompatible xformable" 경고 + transform 미적용 발생.
      → 빈 Xform(wrapper)을 만들고 그 자식에 reference를 넣은 뒤
        wrapper에 transform을 적용하면 충돌 없이 위치/회전 가능.

    구조:  wrapper_path (Xform, 여기에 translate/rotate)
             └ wrapper_path/robot (reference)
    """
    stage = omni.usd.get_context().get_stage()

    # 1) wrapper Xform 생성 (transform 담당)
    wrapper = UsdGeom.Xform.Define(stage, wrapper_path)
    xc = UsdGeom.XformCommonAPI(wrapper)
    xc.SetTranslate((xy[0], xy[1], z))
    xc.SetRotate((0.0, 0.0, yaw_deg), UsdGeom.XformCommonAPI.RotationOrderXYZ)

    # 2) wrapper 자식에 로봇 reference
    ref_path = f"{wrapper_path}/robot"
    add_reference_to_stage(usd_path=usd_path, prim_path=ref_path)

    # 3) 로드 검증
    ref_prim = stage.GetPrimAtPath(ref_path)
    if not ref_prim.IsValid() or len(list(ref_prim.GetChildren())) == 0:
        print(f"  [!] {label}: 로드 실패/비어있음 → {usd_path}")
        return False
    return True


def spawn_jackal_with_arm(cfg: dict):
    """Jackal 1대 + SO-101 manipulator 탑재.
       - arm은 Jackal의 자식으로 종속 (Jackal 옮기면 arm도 함께 이동)
       - Jackal 바디 빨강, 팔 회색
       - 물리 비활성화 (툭 쳐도 안 날아감, 고정)

    계층 구조:
      /World/Jackal_A              (Xform — 위치/회전)
        ├ /World/Jackal_A/robot    (Jackal 본체 reference)
        └ /World/Jackal_A/Arm      (Xform — 등판 위 오프셋)
            └ /World/Jackal_A/Arm/robot  (SO-101 reference)
    """
    name = cfg["name"]
    x, y = cfg["xy"]
    yaw = cfg["yaw_deg"]

    # ── Jackal 본체 (바닥 z=0에 스폰) ─────────────────────────────────────
    jackal_path = f"/World/{name}"
    ok_jackal = add_robot_at(JACKAL_USD, jackal_path, f"{name} (Jackal)",
                             xy=(x, y), z=0.0, yaw_deg=yaw)

    # ── SO-101 Manipulator — Jackal 자식으로 종속 ─────────────────────────
    # arm wrapper를 Jackal_A 하위에 생성 → Jackal과 한 몸으로 묶임.
    # 부모(Jackal)가 이미 (x,y,yaw)를 적용하므로, arm은 로컬 오프셋만 (0,0,DECK_Z).
    arm_path = f"{jackal_path}/Arm"
    ok_arm = add_robot_at(ARM_USD, arm_path, f"{name} (SO-101 arm)",
                          xy=(0.0, 0.0), z=JACKAL_DECK_Z, yaw_deg=0.0)

    # ── 색상: Jackal 바디 빨강, 팔 회색 ───────────────────────────────────
    # arm이 Jackal 자식이므로, Jackal 색칠 시 Arm 경로는 제외해야 함.
    """
    if ok_jackal:
        painted_body = colorize_subtree(
            jackal_path, JACKAL_COLOR,
            skip_keywords=["wheel", "tire", "sensor", "lidar", "camera", "imu"],
            exclude_paths=[arm_path],   # 팔은 빨강 칠 제외
        )
        print(f"    └ Jackal body painted red ({painted_body} meshes)")
    if ok_arm:
        painted_arm = colorize_subtree(arm_path, ARM_COLOR)
        print(f"    └ arm painted gray ({painted_arm} meshes)")
        if painted_arm == 0:
            # SO-101 메쉬를 못 찾음 — 실제 prim 타입 디버그 출력
            _stage = omni.usd.get_context().get_stage()
            _arm_root = _stage.GetPrimAtPath(arm_path)
            print(f"      [debug] arm prim types under {arm_path}:")
            _seen = set()
            for _p in Usd.PrimRange(_arm_root):
                _t = _p.GetTypeName()
                if _t and _t not in _seen:
                    _seen.add(_t)
                    print(f"        type={_t}  e.g. {_p.GetName()}")
    """

    # ── 물리 비활성화 (물리 안 쓰지만 안전상 유지) ─────────────────────────
    n_static = make_static(jackal_path)

    status = "OK" if (ok_jackal and ok_arm) else "PARTIAL/FAIL"
    print(f"[SCENE] {name} @ ({x:.1f}, {y:.1f})  yaw={yaw}°  "
          f"+ SO-101 (child)  [static:{n_static}]  [{status}]")


def load_warehouse():
    """NVIDIA warehouse_multiple_shelves USD 로드.
       선반(shelf/rack), 기둥(pillar), 팔레트(pallet), 벽(wall) 등 실제 구조물 포함.
       spot_factory_inspection.py와 동일 환경 → 논문 일관성 유지.
    """
    ok = _add_ref_checked(ENV_USD, "/World/Factory", "warehouse")
    if ok:
        print("[ENV] warehouse_multiple_shelves loaded → /World/Factory")
    else:
        print("[ENV] warehouse 로드 실패 — 경로 확인 필요")


def hide_overhead_prims(root_path: str = "/World/Factory"):
    """천장/지붕/조명메쉬/보 등 부감(top view)을 가리는 prim 숨김.
       실제 광원(UsdLux Light)은 보존. visual만 끔.
    """
    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        print(f"[ENV] hide_overhead_prims: {root_path} not found")
        return

    keywords = ["ceiling", "roof", "lamp", "beam", "truss",
                "rafter", "girder", "bracket", "slot", "window"]
    hidden = 0
    for prim in Usd.PrimRange(root):
        name = prim.GetName().lower()
        if "Light" in prim.GetTypeName():
            continue
        if any(kw in name for kw in keywords):
            try:
                UsdGeom.Imageable(prim).MakeInvisible()
                hidden += 1
            except Exception:
                pass
    print(f"[ENV] Hidden {hidden} overhead prims (천장/조명/보)")


def deactivate_walls_pillars(root_path: str = "/World/Factory"):
    """Wall, pillar 전부 완전 비활성화 (visual + collision 모두 제거).
       SetActive(False)로 prim 자체를 끔 → stage 트리에서도 비활성으로 표시.
    """
    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        return

    keywords = ["wall", "pillar", "column"]
    # 비활성화할 최상위 prim 경로만 수집 (PrimRange 도중 SetActive 하면 순회 깨짐)
    to_deactivate = []
    for prim in Usd.PrimRange(root):
        name = prim.GetName().lower()
        if any(kw in name for kw in keywords):
            to_deactivate.append(prim.GetPath())

    done = 0
    for path in to_deactivate:
        prim = stage.GetPrimAtPath(path)
        if prim and prim.IsValid():
            prim.SetActive(False)
            done += 1
    print(f"[ENV] Deactivated {done} wall/pillar prims")


def group_shelf_units(root_path: str = "/World/Factory"):
    """흩어진 선반 부속(RackShelf, pallet, smallKLT 등)을
       위치 기준 3개 클러스터로 묶어 계층 정리.

       결과 구조:
         /World/ShelfUnits/
           ├ ShelfUnit_1/  (부속들 reparent)
           ├ ShelfUnit_2/
           └ ShelfUnit_3/

       방법: 부속 prim들의 XY 위치를 K-means(K=3) 유사 방식으로 클러스터링,
             각 클러스터를 ShelfUnit_N Xform 아래로 reparent.
       ※ reparent는 USD에서 prim 이동(Sdf 레이어 편집)이 필요.
    """
    import omni.kit.commands
    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        print("[ENV] group_shelf_units: Factory not found")
        return

    # 1) 선반 부속 prim 수집 (이름 키워드 기반)
    part_keywords = ["rackshelf", "rack_shelf", "pallet", "smallklt",
                     "klt", "shelf", "rack", "bin", "box"]
    exclude = ["wall", "pillar", "floor", "ceiling", "light", "column"]

    parts = []   # (path, (x, y, z))
    for prim in Usd.PrimRange(root):
        name = prim.GetName().lower()
        if any(ex in name for ex in exclude):
            continue
        if not any(kw in name for kw in part_keywords):
            continue
        # 최상위 부속만 (자식 메쉬 중복 수집 방지) — Xform 타입 우선
        if prim.GetTypeName() not in ("Xform", "Scope"):
            continue
        try:
            xc = UsdGeom.XformCommonAPI(prim)
            t = xc.GetXformVectors(Usd.TimeCode.Default())[0]
            parts.append((prim.GetPath(), (float(t[0]), float(t[1]), float(t[2]))))
        except Exception:
            pass

    if len(parts) < 3:
        print(f"[ENV] group_shelf_units: 부속 {len(parts)}개만 발견 — "
              f"그룹화 생략 (이름 키워드 확인 필요)")
        # 인스펙터에서 실제 이름 확인용으로 일부 출력
        for p, pos in parts[:10]:
            print(f"        {p}  @ {tuple(round(v,1) for v in pos)}")
        return

    # 2) X 좌표 기준 3분할 클러스터링 (간단 버전)
    #    창고 선반은 보통 통로 따라 X축으로 나열되므로 X 기준 분할이 자연스러움
    xs = sorted(set(p[1][0] for p in parts))
    x_min, x_max = xs[0], xs[-1]
    span = (x_max - x_min) if x_max > x_min else 1.0

    def cluster_of(pos):
        # X 위치를 0~1로 정규화 후 3분할
        frac = (pos[0] - x_min) / span
        if frac < 1/3:   return 1
        if frac < 2/3:   return 2
        return 3

    # 3) ShelfUnits 부모 Xform 생성
    UsdGeom.Xform.Define(stage, "/World/ShelfUnits")
    for i in (1, 2, 3):
        UsdGeom.Xform.Define(stage, f"/World/ShelfUnits/ShelfUnit_{i}")

    # 4) 각 부속을 해당 클러스터로 reparent (MovePrim 커맨드)
    moved = {1: 0, 2: 0, 3: 0}
    for path, pos in parts:
        c = cluster_of(pos)
        src = str(path)
        dst = f"/World/ShelfUnits/ShelfUnit_{c}/{path.name}"
        try:
            omni.kit.commands.execute("MovePrim", path_from=src, path_to=dst)
            moved[c] += 1
        except Exception as e:
            print(f"  [!] reparent 실패: {src} → {e}")

    print(f"[ENV] Shelf 그룹화 완료 → /World/ShelfUnits/")
    print(f"      ShelfUnit_1: {moved[1]}개, "
          f"ShelfUnit_2: {moved[2]}개, ShelfUnit_3: {moved[3]}개")


def inspect_warehouse_prims(root_path: str = "/World/Factory"):
    """warehouse 내부 shelf/rack/pillar/pallet 등 주요 구조물 개수+샘플 좌표 출력.
       이 출력 보고 어디에 무엇이 있는지 파악 → AGV 위치나 복제 위치 결정에 참고.
    """
    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        return

    keywords = ["shelf", "rack", "rackshelf", "pallet", "smallklt", "klt",
                "bin", "box", "pillar", "wall", "column", "floor", "aisle"]
    counts, samples = {}, {}
    for prim in Usd.PrimRange(root):
        name_lc = prim.GetName().lower()
        for kw in keywords:
            if kw in name_lc:
                counts[kw] = counts.get(kw, 0) + 1
                if kw not in samples:
                    try:
                        xc = UsdGeom.XformCommonAPI(prim)
                        t = xc.GetXformVectors(Usd.TimeCode.Default())[0]
                        samples[kw] = (round(t[0], 1), round(t[1], 1), round(t[2], 1))
                    except Exception:
                        samples[kw] = "?"
                break

    print("\n" + "=" * 60)
    print("  [INSPECT] warehouse 구조물 (복제/배치 참고용)")
    print("=" * 60)
    for kw in keywords:
        if kw in counts:
            print(f"    {kw:8s}: {counts[kw]:3d} 개   (예시 좌표: {samples[kw]})")
    print("=" * 60 + "\n")


def setup_lighting():
    """씬 조명 — DomeLight(균일 환경광) + 상단 DiskLight."""
    stage = omni.usd.get_context().get_stage()
    UsdGeom.Xform.Define(stage, "/World/Lights")

    # 주 조명 (Dome — 따뜻한 흰색 환경광)
    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/DomeLight")
    dome.CreateIntensityAttr(600.0)
    dome.CreateColorAttr(Gf.Vec3f(1.0, 0.98, 0.95))

    # 보조 Disk 조명 (상단에서 아래로, 씬 전체 커버)
    disk = UsdLux.DiskLight.Define(stage, "/World/Lights/DiskLight")
    disk.CreateIntensityAttr(5000.0)
    disk.CreateRadiusAttr(8.0)
    disk_xform = UsdGeom.XformCommonAPI(disk)
    disk_xform.SetTranslate((0.0, 0.0, 18.0))
    disk_xform.SetRotate((180.0, 0.0, 0.0), UsdGeom.XformCommonAPI.RotationOrderXYZ)

    print("[LIGHT] DomeLight + DiskLight 설정 완료")


def hide_grid():
    """viewport 바닥 그리드 숨김.
       그리드는 USD prim이 아니라 viewport 표시 옵션이라 carb 설정으로 끔.
       displayOptions는 비트마スク — grid 비트를 끄는 대신 안전하게
       그리드 전용 설정 키를 False로.
    """
    import carb
    settings = carb.settings.get_settings()
    # 그리드 표시 끄기 (여러 키 시도 — 버전별 키 이름 차이 대응)
    for key in ("/persistent/app/viewport/grid/enabled",
                "/app/viewport/grid/enabled",
                "/persistent/app/viewport/displayOptions/showGrid"):
        try:
            settings.set(key, False)
        except Exception:
            pass
    print("[GRID] viewport grid 숨김")


# ═════════════════════════════════════════════════════════════════════════════
#  메인
# ═════════════════════════════════════════════════════════════════════════════

def main():
    world = World(
        stage_units_in_meters=1.0,
        physics_dt=1.0 / 60.0,
        rendering_dt=1.0 / 60.0,
    )
    # 기본 ground plane은 격자(grid) 평면을 깔아서 캡처에 방해됨 → 추가 안 함.
    # 우리 floor tile이 바닥 역할을 하므로 불필요.

    # ── 씬 구성 ────────────────────────────────────────────────────────────
    hide_grid()        # viewport 그리드 숨김
    setup_lighting()

    # warehouse 구조물 (선반/기둥/팔레트/벽) — --no-warehouse면 생략
    if not _args.no_warehouse:
        load_warehouse()

    spawn_square_floor(half=20.0, tile_size=5.0)
    for cfg in AGV_CONFIGS:
        spawn_jackal_with_arm(cfg)

    # ── 씬 후처리 (reset 없이 — 물리 안 켜면 joint 에러 안 남) ──────────────
    # 그림 캡처용이라 물리 시뮬 불필요. world.reset()/step()은 물리를 켜서
    #   "CreateJoint between static bodies" 에러 + 로봇이 움직임/날아감 유발.
    # → reset 대신 simulation_app.update()로 렌더링만 갱신.
    print("[SCENE] 씬 초기화 중...")
    for _ in range(30):
        simulation_app.update()   # USD 로드/머티리얼 반영 (물리 없음)

    # warehouse 후처리 (prim이 stage에 로드된 뒤 호출)
    if not _args.no_warehouse:
        inspect_warehouse_prims("/World/Factory")
        hide_overhead_prims("/World/Factory")
        deactivate_walls_pillars("/World/Factory")
        group_shelf_units("/World/Factory")

    for _ in range(30):
        simulation_app.update()   # 후처리 반영

    print("[SCENE] Ready. Window is open — adjust camera and capture manually.")

    # ── 창 유지 (물리/캡처 없음, 렌더링만) ──────────────────────────────────
    while simulation_app.is_running():
        simulation_app.update()


if __name__ == "__main__":
    main()
    simulation_app.close()