"""Chess robot reach simulation.

The end of Arm 3 (the Arm1Connector tool on the Joint 3 axis) is moved over a chess square by
inverse kinematics on Joint 1 and Joint 2; Joint 3 keeps the tool yaw fixed; the Z slider
lowers the gripper fingertips until it touches the top of whatever stands on that square.

    python chess_robot_sim.py            # GUI: auto tour (or pick a square by hand)
    python chess_robot_sim.py --direct   # headless self-test of the kinematics
    python chess_robot_sim.py --no-markers   # hide the green "reachable" dots

GUI controls: "Auto tour" slider (1 = tour, 0 = manual); in manual mode set File / Rank and
press "Go to square".
"""
import math
import struct
import sys
import time

import numpy as np

from midnight_scene import *  # noqa: F401,F403  (p, robot, slider, JOINTS, board helpers, ...)

# ----------------------------------------------------------------------------------------
# Geometry (cm). Pivots come from the Fusion joints; the tool tip is on the Joint 3 axis.
# ----------------------------------------------------------------------------------------
P1, P2, P3 = ([v * 100 for v in JOINTS[k]["pivot"][:2]] for k in range(3))
L1, L2 = math.dist(P1, P2), math.dist(P2, P3)
TIP_Z_AT_Q0 = 13.21          # cm: lowest point of the gripper fingers with the slider at 0 (Fusion pose)
BOARD_TOP_CM = board_top * 100
HOVER_ABOVE_BOARD = 8.0      # cm: travel height of the tool tip (clears the tallest piece)
TOUCH_CLEARANCE = 0.4        # cm: how close to the piece top the tip stops
PIECE_HEIGHT = {"P": 3.6, "R": 4.0, "N": 4.4, "B": 4.9, "Q": 4.8, "K": 5.1}   # cm above the board
LIM = [jt["limits"] for jt in JOINTS]


def square_xy_cm(name):
    f, r = "abcdefgh".index(name[0]), int(name[1])
    i = (3 - f) if files_mirrored else (f - 4)
    j = r - 5
    return ((BOARD_ORIGIN[0] + (i + 0.5) * cell) * 100, (BOARD_ORIGIN[1] + (j + 0.5) * cell) * 100)


# Standard starting position -> height of whatever stands on each square.
occupant = {}
for _f, _k in enumerate(back_rank):
    for _file, _kind in (("abcdefgh"[_f], _k),):
        occupant[f"{_file}1"] = occupant[f"{_file}8"] = _kind
for _file in "abcdefgh":
    occupant[f"{_file}2"] = occupant[f"{_file}7"] = "P"


def solve_ik(x, y):
    """All (joint1, joint2) in degrees (about +Z, inside the joint limits) that put the tool at x, y."""
    dx, dy = x - P1[0], y - P1[1]
    c = (dx * dx + dy * dy - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    if abs(c) > 1.0:
        return []
    sols = []
    for sign in (1, -1):                                     # the two elbow configurations
        th2 = sign * math.acos(c)
        phi1 = math.atan2(dy, dx) - math.atan2(L2 * math.sin(th2), L1 + L2 * math.cos(th2))
        t1, t2 = math.degrees(phi1) + 90.0, math.degrees(th2)   # arm points along -Y at 0 deg
        for k1 in (-360, 0, 360):
            for k2 in (-360, 0, 360):
                a1, a2 = t1 + k1, t2 + k2
                if LIM[0][0] <= a1 <= LIM[0][1] and LIM[1][0] <= a2 <= LIM[1][1]:
                    sols.append((a1, a2))
    return sols


def tool_yaw_joint(a1, a2):
    """Joint 3 angle that keeps the tool pointing the way it does in the Fusion pose, clamped."""
    a = -(a1 + a2)
    a = (a + 180.0) % 360.0 - 180.0
    return min(max(a, LIM[2][0]), LIM[2][1])


def slider_for_tip_z(z_cm):
    return min(max((z_cm - TIP_Z_AT_Q0) / 100.0, lo), hi)


def square_config(name, current=None, touch=True):
    """[slider_m, j1, j2, j3, gripper_cm] to put the tool tip over `name` (at touch or hover height); None if out of reach."""
    x, y = square_xy_cm(name)
    sols = solve_ik(x, y)
    if not sols:
        return None
    ref = current[1:3] if current else (0.0, 0.0)
    a1, a2 = min(sols, key=lambda s: abs(s[0] - ref[0]) + abs(s[1] - ref[1]))
    top = PIECE_HEIGHT.get(occupant.get(name), 0.0)
    z = BOARD_TOP_CM + top + TOUCH_CLEARANCE if touch else BOARD_TOP_CM + HOVER_ABOVE_BOARD
    return [slider_for_tip_z(z), a1, a2, tool_yaw_joint(a1, a2), 0.0]


def apply_config(cfg):
    p.resetJointState(robot, slider, cfg[0])
    sync_extra_parts(cfg[1:4], cfg[4] if len(cfg) > 4 else 0.0)


# The slider joint has a default velocity motor in PyBullet; switch it off so poses are exact.
p.setJointMotorControl2(robot, slider, p.VELOCITY_CONTROL, force=0)

ALL_SQUARES = [f"{f}{r}" for r in range(1, 9) for f in "abcdefgh"]
REACHABLE = {s for s in ALL_SQUARES if square_config(s) is not None}

# ----------------------------------------------------------------------------------------
# Self-test (headless): drive the tool to every reachable square and measure where
# the connector really ends up in the PyBullet scene.
# ----------------------------------------------------------------------------------------
racks = {n: b for n, b, _ in extra_parts if n in ("RackA", "RackB")}


def _load_stl_vertices(path):
    raw = open(path, "rb").read()
    n = struct.unpack_from("<I", raw, 80)[0]
    tri = np.frombuffer(raw, dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]),
                        count=n, offset=84)["v"]
    return tri.reshape(-1, 3).astype(float) * 0.001


_RACK_VERTS = {n: _load_stl_vertices(os.path.join(HERE, "meshes", n + ".stl")) for n in racks}


def tool_measure():
    """Where the gripper fingers really are in the scene (mesh vertices through the body poses):
    (midpoint x, midpoint y, lowest z) in cm."""
    centres, zmin = [], 1e9
    for n, body in racks.items():
        pos, orn = p.getBasePositionAndOrientation(body)
        rot = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        w = _RACK_VERTS[n] @ rot.T + np.array(pos)
        centres.append((w.min(axis=0) + w.max(axis=0))[:2] / 2)
        zmin = min(zmin, w[:, 2].min())
    mid = (centres[0] + centres[1]) / 2 * 100
    return (mid[0], mid[1], zmin * 100)


apply_config([0.0, 0.0, 0.0, 0.0, 0.0])
_, _, z0 = tool_measure()
print(f"gripper fingertips at the Fusion pose: z={z0:.2f} cm (expected {TIP_Z_AT_Q0})")


def self_test():
    worst_xy = worst_z = 0.0
    for s in sorted(REACHABLE):
        cfg = square_config(s)
        apply_config(cfg)
        mx, my, mz = tool_measure()
        tx, ty = square_xy_cm(s)
        tz = BOARD_TOP_CM + PIECE_HEIGHT.get(occupant.get(s), 0.0) + TOUCH_CLEARANCE
        worst_xy = max(worst_xy, math.hypot(mx - tx, my - ty))
        worst_z = max(worst_z, abs(mz - tz) if cfg[0] > lo + 1e-9 else 0.0)
    print(f"self-test: {len(REACHABLE)}/64 squares reachable; worst tip error "
          f"{worst_xy * 10:.2f} mm in XY, {worst_z * 10:.2f} mm in Z")


def print_reach_map():
    print("\nReachable squares (# = tool tip can touch it):\n     " + "  ".join("abcdefgh"))
    for r in range(8, 0, -1):
        print(f"  {r}  " + "  ".join("#" if f"{f}{r}" in REACHABLE else "." for f in "abcdefgh"))
    print()


# ----------------------------------------------------------------------------------------
# Motion
# ----------------------------------------------------------------------------------------
DT = 1.0 / 120.0
cur = [0.0, 0.0, 0.0, 0.0, 0.0]
target_marker = None
label_id = None
if not DIRECT:
    target_marker = p.createMultiBody(0, -1, p.createVisualShape(
        p.GEOM_SPHERE, radius=0.006, rgbaColor=(0.9, 0.1, 0.1, 1)), [0, 0, -1])


def move(cfg, seconds):
    """Smoothly interpolate all four joints from the current configuration to `cfg`."""
    global cur
    start = list(cur)
    steps = max(1, int(seconds / DT))
    for k in range(1, steps + 1):
        if not p.isConnected():
            return False
        u = k / steps
        u = u * u * (3 - 2 * u)                      # smoothstep ease in/out
        cur = [a + (b - a) * u for a, b in zip(start, cfg)]
        apply_config(cur)
        if not DIRECT:
            time.sleep(DT)
    return True


def dwell(seconds):
    if not DIRECT:
        t_end = time.time() + seconds
        while time.time() < t_end and p.isConnected():
            time.sleep(DT)


def show_target(name, ok):
    global label_id
    if DIRECT:
        return
    x, y = square_xy_cm(name)
    pos = [x / 100, y / 100, (BOARD_TOP_CM + 9.5) / 100]
    kw = dict(textPosition=pos, textColorRGB=[0.1, 0.5, 0.1] if ok else [0.8, 0.1, 0.1], textSize=1.6)
    text = name if ok else f"{name} (out of reach)"
    label_id = p.addUserDebugText(text, replaceItemUniqueId=label_id, **kw) if label_id is not None \
        else p.addUserDebugText(text, **kw)
    if ok:
        z = (BOARD_TOP_CM + PIECE_HEIGHT.get(occupant.get(name), 0.0) + TOUCH_CLEARANCE) / 100
        p.resetBasePositionAndOrientation(target_marker, [x / 100, y / 100, z], [0, 0, 0, 1])


def visit(name):
    """Raise, swing over the square, lower until the fingertips touch, close/open the gripper, raise."""
    show_target(name, name in REACHABLE)
    if name not in REACHABLE:
        print(f"{name}: out of reach")
        return
    hover = square_config(name, cur, touch=False)
    touch = square_config(name, cur, touch=True)
    up = [slider_for_tip_z(BOARD_TOP_CM + HOVER_ABOVE_BOARD)] + cur[1:]
    ok = move(up, 0.8)                                   # lift to travel height
    ok = ok and move(hover, 1.6)                         # swing over the square
    ok = ok and move(touch, 1.0)                         # descend onto the piece / board
    ok = ok and move(touch[:4] + [1.0], 0.5)             # close the gripper fingers...
    dwell(0.3)
    ok = ok and move(touch, 0.5)                         # ...and open them again
    ok = ok and move(hover, 0.8)                         # back up
    if ok:
        print(f"{name}: j1={touch[1]:7.1f}  j2={touch[2]:7.1f}  j3={touch[3]:6.1f} deg   z={touch[0] * 100:+6.1f} cm")


TOUR = ["e4", "d4", "d3", "e3", "e2", "c3", "f3", "f4", "f5", "e5", "d5", "c4", "b5",
        "a6", "a8", "b7", "g7", "h5", "g4", "h3", "e4"]

print_reach_map()
self_test()

if DIRECT:
    for sq in TOUR[:6]:
        visit(sq)
    p.disconnect()
    sys.exit(0)

if "--no-markers" not in sys.argv:
    dot = p.createVisualShape(p.GEOM_CYLINDER, radius=0.007, length=0.0006, rgbaColor=(0.1, 0.8, 0.2, 0.7))
    for s in REACHABLE:
        x, y = square_xy_cm(s)
        p.createMultiBody(0, -1, dot, [x / 100, y / 100, BOARD_TOP_CM / 100 + 0.0007])

p.resetDebugVisualizerCamera(0.95, 15, -38, [0, -0.08, 0.12])
ui_auto = p.addUserDebugParameter("Auto tour (1=on, 0=manual)", 0, 1, 1)
ui_file = p.addUserDebugParameter("File (0=a .. 7=h)", 0, 7, 4)
ui_rank = p.addUserDebugParameter("Rank (1..8)", 1, 8, 4)
ui_go = p.addUserDebugParameter("Go to square", 1, 0, 0)     # min > max makes it a button
last_go = p.readUserDebugParameter(ui_go)

apply_config(cur)
tour = [s for s in TOUR if s in REACHABLE]
k = 0
while p.isConnected():
    if p.readUserDebugParameter(ui_auto) > 0.5:
        visit(tour[k % len(tour)])
        k += 1
    else:
        go = p.readUserDebugParameter(ui_go)
        if go != last_go:
            last_go = go
            visit("abcdefgh"[int(round(p.readUserDebugParameter(ui_file)))]
                  + str(int(round(p.readUserDebugParameter(ui_rank)))))
        else:
            time.sleep(DT)

if p.isConnected():
    p.disconnect()
