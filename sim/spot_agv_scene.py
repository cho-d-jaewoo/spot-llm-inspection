"""
spot_agv_factory.py  v10 (final)
=================================
Isaac Sim 5.0  —  SPOT RL Walking + Nova Carter AGV

SPOT:
  - isaacsim.robot.policy.examples.robots.SpotFlatTerrainPolicy 사용
  - 공식 NVIDIA 클래스 → device 에러 없음
  - 커스텀 policy/env yaml 경로 로드

Nova Carter:
  - isaacsim.robot.wheeled_robots.robots.WheeledRobot
  - isaacsim.robot.wheeled_robots.controllers.DifferentialController
  - 차동 구동 바퀴 velocity 제어

실행:
  python spot_agv_factory.py
  python spot_agv_factory.py --headless
"""

import argparse
import os
import numpy as np

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true")
args = parser.parse_args()

simulation_app = SimulationApp({"headless": args.headless, "anti_aliasing": 0})

# ── Core imports (SimulationApp 이후) ─────────────────────────────────────────
from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage

from isaacsim.robot.policy.examples.robots.spot import SpotFlatTerrainPolicy
from isaacsim.robot.wheeled_robots.robots import WheeledRobot
from isaacsim.robot.wheeled_robots.controllers.differential_controller import DifferentialController

# ═════════════════════════════════════════════════════════════════════════════
#  설정
# ═════════════════════════════════════════════════════════════════════════════

SPOT_USD = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
    "/Assets/Isaac/5.0/Isaac/Robots/BostonDynamics/spot/spot.usd"
)
NOVA_CARTER_USD = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
    "/Assets/Isaac/5.0/Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd"
)

# ── Repo-relative asset paths ────────────────────────────────────────────────
# Resolved from this file's location so a fresh clone works anywhere. Set
# SPOT_REPO_ROOT to point elsewhere if you keep large assets outside the repo.
from pathlib import Path as _Path
_REPO_ROOT = _Path(os.environ.get("SPOT_REPO_ROOT") or _Path(__file__).resolve().parents[1])

POLICY_PATH = str(_REPO_ROOT / "policies" / "spot_policy.pt")
ENV_YAML    = str(_REPO_ROOT / "policies" / "spot_env.yaml")

# spot_env.yaml 기준: dt=0.002
# rendering_dt = physics_dt 로 설정해야 world.step() 1번 = physics 1번
# spot.py 내부 _decimation(=10)이 policy 실행 주기를 알아서 관리
PHYSICS_DT   = 0.002
RENDERING_DT = 0.002

# SPOT velocity command (vx, vy, wz)
# 훈련 범위: vx [-2.0, 3.0]  vy [-1.0, 1.0]  wz [-2.0, 2.0]
# 이 범위 안에서만 안정적으로 작동
SPOT_CMD = np.array([2.0, 1.0, 0.0])

# AGV 설정
AGV_ROUTES = [
    [(-8.0, -4.0), (-8.0, 4.0), (-2.0, 4.0), (-2.0, -4.0)],
    [( 2.0, -4.0), ( 2.0, 4.0), ( 7.0, 4.0), ( 7.0, -4.0)],
]
WHEEL_RADIUS = 0.14
WHEEL_BASE   = 0.60
AGV_SPEED    = 1.2
AGV_TURN     = 2.0
WP_DIST      = 0.8


# ═════════════════════════════════════════════════════════════════════════════
#  Nova Carter 컨트롤러
# ═════════════════════════════════════════════════════════════════════════════

class CarterController:
    def __init__(self, idx: int, robot: WheeledRobot):
        self.idx        = idx
        self.robot      = robot
        self.ctrl       = DifferentialController(
            name=f"carter_ctrl_{idx}",
            wheel_radius=WHEEL_RADIUS,
            wheel_base=WHEEL_BASE,
        )
        self.route      = [np.array(wp) for wp in AGV_ROUTES[idx]]
        self.wp_idx     = 0
        sx, sy = AGV_ROUTES[idx][0]
        self.x, self.y, self.yaw = float(sx), float(sy), 0.0

    def tick(self):
        p2  = np.array([self.x, self.y])
        tgt = self.route[self.wp_idx]

        if np.linalg.norm(p2 - tgt) < WP_DIST:
            self.wp_idx = (self.wp_idx + 1) % len(self.route)
            tgt = self.route[self.wp_idx]

        tgt_yaw = float(np.arctan2(tgt[1] - p2[1], tgt[0] - p2[0]))
        err     = (tgt_yaw - self.yaw + np.pi) % (2 * np.pi) - np.pi
        wz      = float(np.clip(AGV_TURN * err, -3.0, 3.0))
        vx      = float(AGV_SPEED * max(0.0, 1.0 - abs(err) / 1.2))

        # 내부 상태 적분 (robot.data 읽기 없이)
        self.x   += vx * np.cos(self.yaw) * PHYSICS_DT
        self.y   += vx * np.sin(self.yaw) * PHYSICS_DT
        self.yaw += wz * PHYSICS_DT

        # DifferentialController → 바퀴 velocity action
        action = self.ctrl.forward(command=np.array([vx, wz]))
        self.robot.apply_wheel_actions(action)


# ═════════════════════════════════════════════════════════════════════════════
#  메인
# ═════════════════════════════════════════════════════════════════════════════

def main():
    world = World(
        stage_units_in_meters=1.0,
        physics_dt=PHYSICS_DT,
        rendering_dt=RENDERING_DT,
    )
    world.scene.add_default_ground_plane()

    # ── SPOT 스폰 ─────────────────────────────────────────────────────────────
    spot = SpotFlatTerrainPolicy(
        prim_path="/World/SPOT",
        name="spot",
        usd_path=SPOT_USD,
        position=np.array([0.0, 0.0, 0.4]),
    )
    # 공식 클래스는 기본 asset 경로에서 policy 로드 → 커스텀 경로로 덮어쓰기
    spot.load_policy(POLICY_PATH, ENV_YAML)
    print(f"[SPAWN] SPOT at (0, 0, 0.65)  |  policy: {POLICY_PATH}")

    # ── Nova Carter 스폰 ──────────────────────────────────────────────────────
    agv_robots = []
    for i, route in enumerate(AGV_ROUTES):
        sx, sy = route[0]
        robot = WheeledRobot(
            prim_path=f"/World/NovaCarter_{i}",
            name=f"nova_carter_{i}",
            wheel_dof_names=["joint_wheel_left", "joint_wheel_right"],
            create_robot=True,
            usd_path=NOVA_CARTER_USD,
            position=np.array([sx, sy, 0.3]),
        )
        world.scene.add(robot)
        agv_robots.append(robot)
        print(f"[SPAWN] NovaCarter-{i} at ({sx}, {sy})")

    world.reset()

    # ── 컨트롤러 초기화 ───────────────────────────────────────────────────────
    spot.initialize()
    spot.post_reset()

    # ★ 핵심: initialize()가 stiffness=60 gain을 세팅하면서 zero position에서
    #         default_pos로 당기는 힘이 폭발적으로 발생 → 공중으로 튐
    #         해결: 관절을 default_pos로 먼저 직접 세팅 후 안정화 스텝 실행
    from isaacsim.core.utils.types import ArticulationAction
    spot.robot.set_joint_positions(spot.default_pos)
    spot.robot.set_joint_velocities(spot.default_vel)
    for _ in range(200):  # 0.4초 안정화 (착지 완료)
        world.step(render=False)

    agv_ctrls = [CarterController(i, agv_robots[i]) for i in range(len(agv_robots))]

    print("\n" + "=" * 60)
    print("  SPOT RL Walking + Nova Carter AGV Fleet")
    print(f"  SPOT  : SpotFlatTerrainPolicy  |  cmd vx={SPOT_CMD[0]} m/s")
    print(f"  AGVs  : {len(agv_ctrls)} × Nova Carter")
    print("=" * 60 + "\n")

    step = 0
    while simulation_app.is_running():
        # 액션 먼저 세팅 → 그 다음 물리 적용 (Isaac Sim 올바른 순서)
        spot.forward(dt=PHYSICS_DT, command=SPOT_CMD)
        for ctrl in agv_ctrls:
            ctrl.tick()
        world.step(render=True)

        step += 1
        if step % 500 == 0:
            t = step * PHYSICS_DT
            print(f"[t={t:.1f}s]  "
                  + "  |  ".join(
                      f"AGV{c.idx}=({c.x:.1f},{c.y:.1f})"
                      for c in agv_ctrls))


if __name__ == "__main__":
    main()
    simulation_app.close()