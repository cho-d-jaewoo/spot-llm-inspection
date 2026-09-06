"""
Isaac Sim 5.0 / Isaac Lab 0.54.3 / Windows 11
"""

import argparse
import glob
import os
import numpy as np
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="AGV Factory — Anymal C")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim import SimulationContext
from isaaclab_assets.robots.anymal import ANYMAL_C_CFG

# ── Isaac Sim extension cache ────────────────────────────────────────────────
# Warehouse USD assets ship inside the isaacsim package. We locate them via the
# installed isaacsim module rather than a hardcoded conda path; override with
# ISAACSIM_EXTSCACHE if your install lives elsewhere.
def _default_extscache() -> str:
    try:
        import isaacsim
        return os.path.join(os.path.dirname(isaacsim.__file__), "extscache")
    except Exception:
        return ""


EXTSCACHE = os.environ.get("ISAACSIM_EXTSCACHE") or _default_extscache()

def find_glob(pattern: str):
    hits = glob.glob(os.path.join(EXTSCACHE, pattern), recursive=False)
    return hits[0] if hits else None

WAREHOUSE_USD = find_glob(
    r"isaacsim.replicator.caption.core-*\data\isaacsim_full_warehouse.usda"
)
print(f"[ASSET] Warehouse : {WAREHOUSE_USD or 'not found — using simple walls'}\n")

# ── 경로 레이아웃 ──────────────────────────────────────────────────────────────
AGV_ROUTES = [
    [(-6.0, -2.0), (-6.0, 4.0), (-2.0, 4.0), (-2.0, -2.0)],
    [(-0.5, -2.0), (-0.5, 4.0), ( 3.5, 4.0), ( 3.5, -2.0)],
    [( 5.0, -2.0), ( 5.0, 4.0), ( 8.0, 4.0), ( 8.0, -2.0)],
]

OBSTACLE_POS  = np.array([-0.5, 1.5, 0.0])
FAULT_DIST    = 1.8
WP_REACH_DIST = 0.6
FWD_SPEED     = 0.8
TURN_GAIN     = 1.5


def build_scene():
    sim_utils.DomeLightCfg(intensity=1800.0, color=(1.0, 0.97, 0.90)).func(
        "/World/Dome", sim_utils.DomeLightCfg(intensity=1800.0))
    sim_utils.GroundPlaneCfg().func("/World/Ground", sim_utils.GroundPlaneCfg())

    if WAREHOUSE_USD and os.path.exists(WAREHOUSE_USD):
        wh = sim_utils.UsdFileCfg(usd_path=WAREHOUSE_USD)
        wh.func("/World/Warehouse", wh,
                translation=(0.0, 1.0, 0.0),
                scale=(0.01, 0.01, 0.01))
        print("[SCENE] Full warehouse USD loaded")
    else:
        wall_mat = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.55))
        for tag, sz, pos in [
            ("N", (22.0, 0.3, 3.0), ( 0.0,  6.5, 1.5)),
            ("S", (22.0, 0.3, 3.0), ( 0.0, -3.5, 1.5)),
            ("E", (0.3, 10.0, 3.0), (11.0,  1.5, 1.5)),
            ("W", (0.3, 10.0, 3.0), (-11.0, 1.5, 1.5)),
        ]:
            c = sim_utils.CuboidCfg(
                size=sz, visual_material=wall_mat,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg())
            c.func(f"/World/Wall{tag}", c, translation=pos)

        shelf_mat = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.40, 0.25))
        for i, (sx, sy) in enumerate([(-9.0, -1.0), (-9.0, 2.0), (-9.0, 5.0),
                                        ( 9.5, -1.0), ( 9.5, 2.0), ( 9.5, 5.0)]):
            c = sim_utils.CuboidCfg(
                size=(1.0, 0.4, 2.0), visual_material=shelf_mat,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg())
            c.func(f"/World/Shelf{i}", c, translation=(sx, sy, 1.0))
        print("[SCENE] Minimal warehouse built")

    obs = sim_utils.CuboidCfg(
        size=(0.6, 0.6, 0.6),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.1, 0.1)),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        collision_props=sim_utils.CollisionPropertiesCfg())
    obs.func("/World/Obstacle", obs,
             translation=(OBSTACLE_POS[0], OBSTACLE_POS[1], 0.3))

    for ri, route in enumerate(AGV_ROUTES):
        for wi, (wx, wy) in enumerate(route):
            m = sim_utils.CuboidCfg(
                size=(0.15, 0.15, 0.02),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.1, 0.3, 1.0)),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg())
            m.func(f"/World/WP{ri}_{wi}", m, translation=(wx, wy, 0.01))


def spawn_agvs(n: int) -> list[Articulation]:
    robots = []
    for i in range(n):
        sx, sy = AGV_ROUTES[i][0]
        cfg = ANYMAL_C_CFG.replace(prim_path=f"/World/AGV{i}")
        cfg = cfg.replace(
            init_state=ArticulationCfg.InitialStateCfg(pos=(sx, sy, 0.6))
        )
        robots.append(Articulation(cfg))
        print(f"[AGV{i}] Anymal C spawned at ({sx:.1f}, {sy:.1f})")
    return robots


def quat_yaw(q: np.ndarray) -> float:
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))


class AGVCtrl:
    def __init__(self, i: int, robot: Articulation):
        self.i       = i
        self.robot   = robot
        self.route   = [np.array(wp) for wp in AGV_ROUTES[i]]
        self.wp      = 0
        self.faulted = False
        self._logged = False

    def pos_yaw(self):
        s = self.robot.data.root_state_w[0]
        return s[:3].cpu().numpy(), quat_yaw(s[3:7].cpu().numpy())

    def tick(self):
        pos, yaw = self.pos_yaw()
        p2 = pos[:2]

        d = float(np.linalg.norm(p2 - OBSTACLE_POS[:2]))
        if d < FAULT_DIST and not self.faulted:
            self.faulted = True

        if self.faulted:
            if not self._logged:
                self._logged = True
                print("\n" + "━"*64)
                print(f"  🚨  FAULT  —  AGV-{self.i} STOPPED")
                print(f"      Position  : ({p2[0]:.2f}, {p2[1]:.2f})")
                print(f"      Obstacle  : ({OBSTACLE_POS[0]:.2f}, {OBSTACLE_POS[1]:.2f})")
                print(f"      Distance  : {d:.2f} m  (threshold {FAULT_DIST} m)")
                print()
                print("  [ROS2]  pub /agv_1/fault  →  OBSTACLE_DETECTED")
                print("  [SPOT]  navigating to fault position …")
                print("  [VLM]   image → scene analysis → action_id selected")
                print("  [PLAN]  fleet re-planning issued …")
                print("━"*64 + "\n")
            self._apply_vel(0.0, 0.0, 0.0)
            return

        tgt = self.route[self.wp]
        dw  = float(np.linalg.norm(p2 - tgt))
        if dw < WP_REACH_DIST:
            self.wp = (self.wp + 1) % len(self.route)
            self._apply_vel(0.0, 0.0, 0.0)
            return

        tgt_yaw = np.arctan2(tgt[1] - p2[1], tgt[0] - p2[0])
        err     = (tgt_yaw - yaw + np.pi) % (2*np.pi) - np.pi
        wz      = float(np.clip(TURN_GAIN * err, -2.0, 2.0))
        vx      = FWD_SPEED * max(0.0, 1.0 - abs(err) / 1.2)
        self._apply_vel(vx, 0.0, wz)

    def _apply_vel(self, vx: float, vy: float, wz: float):
        vel = torch.zeros(1, 6, device=self.robot.device)
        _, yaw = self.pos_yaw()
        vel[0, 0] = vx * np.cos(yaw) - vy * np.sin(yaw)
        vel[0, 1] = vx * np.sin(yaw) + vy * np.cos(yaw)
        vel[0, 5] = wz
        self.robot.write_root_velocity_to_sim(vel)
        self.robot.write_data_to_sim()


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, render_interval=4,
                                       gravity=(0.0, 0.0, -9.81))
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=(0.0, -14.0, 12.0), target=(0.0, 1.0, 0.0))

    build_scene()
    robots = spawn_agvs(3)
    sim.reset()

    ctrls = [AGVCtrl(i, robots[i]) for i in range(3)]

    print(f"\n[SIM] Anymal C x3 실행 중 | "
          f"빨간 박스 ({OBSTACLE_POS[0]}, {OBSTACLE_POS[1]}) → Robot-1 FAULT 예정\n")

    step = 0
    while simulation_app.is_running():
        for c in ctrls:
            c.tick()
        sim.step()
        for r in robots:
            r.update(sim.get_physics_dt())
        step += 1

        if step % 500 == 0:
            parts = []
            for c in ctrls:
                p, _ = c.pos_yaw()
                st = "FAULT" if c.faulted else f"WP{c.wp}"
                parts.append(f"R{c.i}=({p[0]:.1f},{p[1]:.1f})[{st}]")
            print(f"[t={step*0.005:.1f}s]  " + "  ".join(parts))


if __name__ == "__main__":
    main()
    simulation_app.close()