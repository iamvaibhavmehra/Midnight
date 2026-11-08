"""Free-running demo: Z slider and the three revolute joints oscillate (or drive them by hand).
The scene itself lives in midnight_scene.py."""
import math
import time

from midnight_scene import *  # noqa: F401,F403

# Auto demo: all four joints oscillate (kept inside their Fusion limits). In the GUI you can
# switch it off with the "Auto demo" slider and drive every joint by hand.
mid, amp = (lo + hi) / 2, (hi - lo) / 2

if not DIRECT:
    p.resetDebugVisualizerCamera(1.1, 40, -25, [0, -0.05, 0.25])
    ui_auto = p.addUserDebugParameter("Auto demo (1=on, 0=manual)", 0, 1, 1)
    ui_z = p.addUserDebugParameter("Z slider (m)", lo, hi, 0.0)
    ui_grip = p.addUserDebugParameter("Gripper close (cm)", 0, GRIP_MAX_CM, 0.0)
    ui_j = [p.addUserDebugParameter(f"{jt['name']} (deg)", *jt["limits"], 0.0) for jt in JOINTS]

steps = 240 * 16 if DIRECT else None
t = 0.0
dt = 1.0 / 240.0
n = 0
while p.isConnected() and (steps is None or n < steps):
    if DIRECT or p.readUserDebugParameter(ui_auto) > 0.5:
        target = mid + amp * math.sin(2 * math.pi * t / 6.0)
        grip = GRIP_MAX_CM * (1 - math.cos(2 * math.pi * t / 2.5)) / 2
        angles = [(d[0] + d[1]) / 2 + (d[1] - d[0]) / 2 * -math.cos(2 * math.pi * t / d[2])
                  for d in (jt["demo"] for jt in JOINTS)]
    else:
        target = p.readUserDebugParameter(ui_z)
        angles = [p.readUserDebugParameter(u) for u in ui_j]
        grip = p.readUserDebugParameter(ui_grip)
    p.setJointMotorControl2(
        robot, slider, p.POSITION_CONTROL,
        targetPosition=target, force=1000, maxVelocity=0.2,
    )
    p.stepSimulation()
    sync_extra_parts(angles, grip)
    if n % 240 == 0:
        print(f"t={t:5.2f} z={p.getJointState(robot, slider)[0]:+.3f} m  "
              + "  ".join(f"j{i + 1}={a:6.1f}" for i, a in enumerate(angles)))
    if not DIRECT:
        time.sleep(dt)
    t += dt
    n += 1

if p.isConnected():
    p.disconnect()
