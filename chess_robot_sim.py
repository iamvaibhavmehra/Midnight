"""Chess robot pick-and-place simulation (with contact physics).

* The chess pieces are dynamic rigid bodies (mass, friction, compound collision shapes).
* The gripper (housing + two finger racks) is a real physical body: the housing follows the arm
  through a strong constraint and each finger is a prismatic joint driven by a force-limited
  motor, so the pieces are held by friction, not glued.
* The arm itself is positioned by inverse kinematics on Joint 1 / Joint 2 (Joint 3 keeps the
  gripper yaw, the Z slider sets the height).

    python chess_robot_sim.py              # GUI: plays a few moves, then manual control
    python chess_robot_sim.py --direct     # headless run + checks
    python chess_robot_sim.py --no-markers # hide the green "reachable" dots

GUI controls: set From/To file + rank and press "Move piece"; "Reset board" puts every
piece back; "Auto moves" (1 = play the demo moves).
"""
import math
import os
import struct
import sys
import time

os.environ["MIDNIGHT_PHYSICS"] = "1"      # must be set before midnight_scene is imported

import numpy as np

from midnight_scene import *  # noqa: F401,F403  (p, robot, slider, JOINTS, part_pose, piece_bodies, ...)

# ----------------------------------------------------------------------------------------
# Geometry (cm). Pivots come from the Fusion joints; the tool point is the middle of the fingers.
# ----------------------------------------------------------------------------------------
P1, P2, P3 = ([v * 100 for v in JOINTS[k]["pivot"][:2]] for k in range(3))
L1, L2 = math.dist(P1, P2), math.dist(P2, P3)
TIP_Z_AT_Q0 = 13.21          # cm: lowest point of the gripper fingers with the slider at 0 (Fusion pose)
BOARD_TOP_CM = board_top * 100
HOVER_ABOVE_BOARD = 8.0      # cm: travel height of the fingertips when empty-handed
CARRY_ABOVE_BOARD = 8.0      # cm: travel height of the fingertips while carrying a piece
GRASP_TIP_ABOVE_BOARD = 0.9  # cm: fingertips this high above the board when gripping (clears the bases)
PLACE_EXTRA = 0.15           # cm: lower the held piece to this far above the board before letting go
LIM = [jt["limits"] for jt in JOINTS]
GRIP_FORCE = 40.0            # N per finger
GRIP_SPEED = 0.04            # m/s finger speed


def square_xy_cm(name):
    f, r = "abcdefgh".index(name[0]), int(name[1])
    i = (3 - f) if files_mirrored else (f - 4)
    j = r - 5
    return ((BOARD_ORIGIN[0] + (i + 0.5) * cell) * 100, (BOARD_ORIGIN[1] + (j + 0.5) * cell) * 100)


# Who stands where (standard start). piece_at[square] -> body id.
piece_at, piece_kind, start_pose = {}, {}, {}
for (_kind, _color, _f), _body in piece_bodies.items():
    _file = "abcdefgh"[_f]
    _rank = {("white", True): 1, ("white", False): 2, ("black", True): 8, ("black", False): 7}[
        (_color, _kind != "P")]
    piece_at[f"{_file}{_rank}"] = _body
    piece_kind[_body] = _kind
    start_pose[_body] = (f"{_file}{_rank}", p.getBasePositionAndOrientation(_body))


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
    """Joint 3 angle that keeps the gripper pointing the way it does in the Fusion pose, clamped."""
    a = -(a1 + a2)
    a = (a + 180.0) % 360.0 - 180.0
    return min(max(a, LIM[2][0]), LIM[2][1])


# Jaw plates in plan view (cm, relative to the tool point, along/across the closing axis).
JAW_ALONG = (2.02, 3.52)
JAW_ACROSS = 2.5
FUSION_AXIS_DEG = 50.0       # closing axis direction in the world at the Fusion pose
PIECE_RADIUS_AT_JAW = {"P": 1.25, "R": 1.55, "N": 1.2, "B": 1.4, "Q": 1.7, "K": 1.7}   # cm, before scaling


def jaw_clearance(phi_deg, tool_xy, ignore=()):
    """Smallest gap (cm) between the jaw plates and the neighbouring pieces if the closing axis points
    along phi_deg. Negative = the jaws would run into a neighbour."""
    u = (math.cos(math.radians(phi_deg)), math.sin(math.radians(phi_deg)))
    v = (-u[1], u[0])
    best = 99.0
    for sq_name, body in piece_at.items():
        if sq_name in ignore:
            continue
        nx, ny = square_xy_cm(sq_name)
        dx, dy = nx - tool_xy[0], ny - tool_xy[1]
        if dx * dx + dy * dy > 8.5 ** 2:
            continue
        a, b = abs(dx * u[0] + dy * u[1]), abs(dx * v[0] + dy * v[1])
        gap_along = max(JAW_ALONG[0] - a, a - JAW_ALONG[1], 0.0)
        gap_across = max(b - JAW_ACROSS, 0.0)
        r = PIECE_RADIUS_AT_JAW[piece_kind[body]] * PIECE_RADIAL_SCALE
        best = min(best, math.hypot(gap_along, gap_across) - r)
    return best


def best_j3(a1, a2, tool_xy, ignore=()):
    """Joint 3 angle (inside its limits) that turns the jaws to where they clear the neighbours best;
    among equally good angles prefer the one that keeps the gripper yaw."""
    keep = tool_yaw_joint(a1, a2)
    cands = [LIM[2][0] + k * 0.5 for k in range(int((LIM[2][1] - LIM[2][0]) / 0.5) + 1)] + [keep]
    return max(cands, key=lambda j3: (min(jaw_clearance(FUSION_AXIS_DEG + a1 + a2 + j3, tool_xy, ignore), 0.6),
                                      -abs(j3 - keep)))


def slider_for_tip_z(z_cm):
    return min(max((z_cm - TIP_Z_AT_Q0) / 100.0, lo), hi)


def config_over(name, tip_z_cm, current=None, grip=0.0, ignore=()):
    """[slider_m, j1, j2, j3, gripper_cm] with the fingertips over square `name` at height tip_z_cm.
    Joint 3 is chosen so the jaw plates stay clear of neighbouring pieces (squares in `ignore`
    are not counted, e.g. the piece being picked up)."""
    x, y = square_xy_cm(name)
    sols = solve_ik(x, y)
    if not sols:
        return None
    ref = current[1:3] if current else (0.0, 0.0)
    a1, a2 = min(sols, key=lambda s: abs(s[0] - ref[0]) + abs(s[1] - ref[1]))
    return [slider_for_tip_z(tip_z_cm), a1, a2, best_j3(a1, a2, (x, y), ignore), grip]


ALL_SQUARES = [f"{f}{r}" for r in range(1, 9) for f in "abcdefgh"]
REACHABLE = {s for s in ALL_SQUARES if config_over(s, BOARD_TOP_CM + 1.0) is not None}

# ----------------------------------------------------------------------------------------
# Physics set-up
# ----------------------------------------------------------------------------------------
p.setPhysicsEngineParameter(fixedTimeStep=1.0 / 240.0, numSolverIterations=200,
                            numSubSteps=1, contactBreakingThreshold=0.001)
# The slider joint is positioned by the script; keep it rigidly at its target.
p.setJointMotorControl2(robot, slider, p.POSITION_CONTROL, targetPosition=0.0, force=1e5)


def _load_stl_vertices(path):
    raw = open(path, "rb").read()
    n = struct.unpack_from("<I", raw, 80)[0]
    tri = np.frombuffer(raw, dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]),
                        count=n, offset=84)["v"]
    return tri.reshape(-1, 3).astype(float) * 0.001


_RACK_VERTS = {n: _load_stl_vertices(os.path.join(HERE, "meshes", n + ".stl")) for n in ("RackA", "RackB")}

JAW_Z_MAX_CM = 16.9   # fingers: only the jaw plate (below the housing, z < this at the Fusion pose) collides


def _rel(parent, child):
    inv = p.invertTransform(*parent)
    return p.multiplyTransforms(inv[0], inv[1], *child)


def build_gripper():
    """Physical gripper: housing (base) + two finger racks on prismatic joints along the closing axis."""
    poses = {n: pose for n, _, pose in extra_parts if n in PHYSICAL_PARTS}
    mesh = lambda n: os.path.join(HERE, "meshes", n + ".stl")
    col = lambda n: p.createCollisionShape(p.GEOM_MESH, fileName=mesh(n), meshScale=[0.001] * 3)

    def jaw_col(n):
        """Convex hull of just the jaw plate: the rack's own convex hull would also fill the space
        under the housing where the other finger sits."""
        rot = np.array(p.getMatrixFromQuaternion(poses[n][1])).reshape(3, 3)
        z_world = (_RACK_VERTS[n] @ rot.T)[:, 2] + poses[n][0][2]
        jaw = _RACK_VERTS[n][z_world * 100 < JAW_Z_MAX_CM]
        return p.createCollisionShape(p.GEOM_MESH, vertices=jaw.tolist())
    vis = lambda n, c: p.createVisualShape(p.GEOM_MESH, fileName=mesh(n), meshScale=[0.001] * 3, rgbaColor=c)
    house = poses["GripperHousing"]
    link_pos, link_orn, axes, cols, viss = [], [], [], [], []
    for n in ("RackA", "RackB"):
        rp, ro = _rel(house, poses[n])
        link_pos.append(rp); link_orn.append(ro)
        dx, dy = RACK_CLOSE_DIR[n]
        inv_q = p.invertTransform([0, 0, 0], poses[n][1])[1]
        axes.append(list(p.multiplyTransforms([0, 0, 0], inv_q, [dx, dy, 0], [0, 0, 0, 1])[0]))
        cols.append(jaw_col(n)); viss.append(vis(n, (0.12, 0.12, 0.12, 1)))
    g = p.createMultiBody(
        baseMass=1.0, baseCollisionShapeIndex=col("GripperHousing"),
        baseVisualShapeIndex=vis("GripperHousing", (0.07, 0.07, 0.07, 1)),
        basePosition=house[0], baseOrientation=house[1],
        linkMasses=[0.08, 0.08], linkCollisionShapeIndices=cols, linkVisualShapeIndices=viss,
        linkPositions=link_pos, linkOrientations=link_orn,
        linkInertialFramePositions=[[0, 0, 0]] * 2, linkInertialFrameOrientations=[[0, 0, 0, 1]] * 2,
        linkParentIndices=[0, 0], linkJointTypes=[p.JOINT_PRISMATIC] * 2, linkJointAxis=axes)
    for j in (0, 1):
        p.changeDynamics(g, j, jointLowerLimit=0.0, jointUpperLimit=GRIP_MAX_CM / 100.0,
                         lateralFriction=2.0, spinningFriction=0.01, linearDamping=0.2)
        p.setJointMotorControl2(g, j, p.POSITION_CONTROL, targetPosition=0.0, force=GRIP_FORCE)
    p.changeDynamics(g, -1, lateralFriction=0.6, linearDamping=0.5, angularDamping=0.5)
    for gl in (-1, 0, 1):                                   # the arm is positioned kinematically:
        for rl in range(-1, p.getNumJoints(robot)):         # no collisions between it and the gripper
            p.setCollisionFilterPair(g, robot, gl, rl, 0)
    cid = p.createConstraint(g, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], house[0],
                             childFrameOrientation=house[1])
    return g, cid


gripper, gripper_cid = build_gripper()
cur = [0.0, 0.0, 0.0, 0.0, 0.0]


def tool_measure():
    """Where the finger meshes are according to the kinematic chain: (midpoint x, y, lowest z) in cm."""
    centres, zmin = [], 1e9
    for n in ("RackA", "RackB"):
        pos, orn = part_pose[n]
        rot = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        w = _RACK_VERTS[n] @ rot.T + np.array(pos)
        centres.append((w.min(axis=0) + w.max(axis=0))[:2] / 2)
        zmin = min(zmin, w[:, 2].min())
    mid = (centres[0] + centres[1]) / 2 * 100
    return (mid[0], mid[1], zmin * 100)


def drive(cfg):
    """Command the arm to cfg and advance the physics one step (finger motors keep their last command)."""
    global cur
    cur = list(cfg)
    p.resetJointState(robot, slider, cfg[0])
    p.setJointMotorControl2(robot, slider, p.POSITION_CONTROL, targetPosition=cfg[0], force=1e5)
    sync_extra_parts(cfg[1:4])
    pos, orn = part_pose["GripperHousing"]
    p.changeConstraint(gripper_cid, jointChildPivot=pos, jointChildFrameOrientation=orn, maxForce=8000)
    p.stepSimulation()


def set_grip(cm):
    """Target finger closure in cm (0 = open, 1.5 = fully closed); the motors push with GRIP_FORCE."""
    for j in (0, 1):
        p.setJointMotorControl2(gripper, j, p.POSITION_CONTROL, targetPosition=cm / 100.0,
                                force=GRIP_FORCE, maxVelocity=GRIP_SPEED)


DT = 1.0 / 240.0


def _tick():
    if not DIRECT:
        time.sleep(DT)


def move(cfg, seconds):
    """Smoothly interpolate slider + joints."""
    start = list(cur)
    steps = max(1, int(seconds / DT))
    for k in range(1, steps + 1):
        if not p.isConnected():
            return False
        u = k / steps
        u = u * u * (3 - 2 * u)                      # smoothstep ease in/out
        drive([a + (b - a) * u for a, b in zip(start[:4], cfg[:4])] + [cur[4]])
        _tick()
    return True


def wait(seconds):
    for _ in range(int(seconds / DT)):
        if not p.isConnected():
            return
        drive(cur)
        _tick()


# ----------------------------------------------------------------------------------------
# Pick and place
# ----------------------------------------------------------------------------------------
label_id = None


def show_label(text, name, color=(0.1, 0.5, 0.1)):
    global label_id
    if DIRECT:
        return
    x, y = square_xy_cm(name)
    kw = dict(textPosition=[x / 100, y / 100, (BOARD_TOP_CM + 10.0) / 100], textColorRGB=list(color), textSize=1.6)
    label_id = p.addUserDebugText(text, replaceItemUniqueId=label_id, **kw) if label_id is not None \
        else p.addUserDebugText(text, **kw)


def piece_height_above_board(body):
    # getBasePositionAndOrientation reports the centre of mass, PIECE_COM_HEIGHT above the base
    return (p.getBasePositionAndOrientation(body)[0][2] - PIECE_COM_HEIGHT) * 100 - BOARD_TOP_CM


def pick_and_place(src, dst):
    """Pick the piece on `src` with the gripper and put it down on `dst`. Returns True on success."""
    if src not in piece_at:
        print(f"{src}: no piece there"); return False
    if dst in piece_at:
        print(f"{dst}: square occupied"); return False
    if src not in REACHABLE or dst not in REACHABLE:
        print(f"{src}->{dst}: out of reach"); return False
    body = piece_at[src]
    kind = piece_kind[body]
    before = {b: p.getBasePositionAndOrientation(b)[0] for b in piece_kind if b != body}
    hover_z = BOARD_TOP_CM + HOVER_ABOVE_BOARD
    grasp_z = BOARD_TOP_CM + GRASP_TIP_ABOVE_BOARD
    print(f"\n{src} -> {dst}  ({kind})")

    show_label(f"pick {src}", src)
    set_grip(0.0)
    ok = move([slider_for_tip_z(hover_z)] + cur[1:4], 0.8)                      # lift to travel height
    ok = ok and move(config_over(src, hover_z, cur, ignore=(src,)), 1.6)                       # over the piece
    ok = ok and move(config_over(src, grasp_z, cur, ignore=(src,)), 1.1)                       # fingers down around it
    if not ok:
        return False
    wait(0.2)
    set_grip(GRIP_MAX_CM)                                                       # squeeze
    wait(0.9)
    closed = sum(p.getJointState(gripper, j)[0] for j in (0, 1)) * 100
    print(f"  fingers closed {closed:.2f} cm in total (max {2 * GRIP_MAX_CM:.1f}) around the {kind}")
    ok = move(config_over(src, BOARD_TOP_CM + CARRY_ABOVE_BOARD, cur, ignore=(src,)), 1.2)     # lift it
    z_lift = piece_height_above_board(body)
    held = z_lift > 3.0
    print(f"  piece is {z_lift:.1f} cm above the board -> {'GRIPPED' if held else 'DROPPED'}")
    if not held:
        set_grip(0.0); wait(0.5)
        return False

    show_label(f"place {dst}", dst)
    ok = ok and move(config_over(dst, BOARD_TOP_CM + CARRY_ABOVE_BOARD, cur, ignore=(src,)), 2.4)    # carry over
    ok = ok and move(config_over(dst, grasp_z + PLACE_EXTRA, cur, ignore=(src,)), 1.3)               # lower onto the board
    wait(0.3)
    set_grip(0.0)                                                                     # let go
    wait(0.8)
    ok = ok and move(config_over(dst, hover_z, cur, ignore=(src,)), 1.0)                             # back up
    wait(0.4)

    piece_at[dst] = piece_at.pop(src)
    tx, ty = square_xy_cm(dst)
    (px, py, pz), orn = p.getBasePositionAndOrientation(body)
    err_mm = math.hypot(px * 100 - tx, py * 100 - ty) * 10
    tilt = math.degrees(2 * math.asin(min(1.0, math.hypot(orn[0], orn[1]))))
    moved = max(math.dist(before[b], p.getBasePositionAndOrientation(b)[0]) for b in before) * 1000
    print(f"  bystander pieces moved by at most {moved:.1f} mm")
    print(f"  placed on {dst}: {err_mm:.1f} mm from the square centre, tilt {tilt:.1f} deg, "
          f"{piece_height_above_board(body):+.2f} cm above the board")
    return ok


def reset_board():
    """Put every piece back on its starting square."""
    piece_at.clear()
    for body, (sq_name, (pos, orn)) in start_pose.items():
        p.resetBasePositionAndOrientation(body, pos, orn)
        p.resetBaseVelocity(body, [0, 0, 0], [0, 0, 0])
        piece_at[sq_name] = body
    wait(0.5)


DEMO_MOVES = [("e2", "e4"), ("d2", "d4"), ("b7", "b5"), ("g7", "g5")]


def print_reach_map():
    print("\nReachable squares (# = fingers can get there):\n     " + "  ".join("abcdefgh"))
    for r in range(8, 0, -1):
        print(f"  {r}  " + "  ".join("#" if f"{f}{r}" in REACHABLE else "." for f in "abcdefgh"))
    print()


def self_test():
    """Kinematic check: the finger meshes end up over each reachable square at the intended height."""
    worst = 0.0
    for s in sorted(REACHABLE):
        cfg = config_over(s, BOARD_TOP_CM + GRASP_TIP_ABOVE_BOARD)
        p.resetJointState(robot, slider, cfg[0]); sync_extra_parts(cfg[1:4])
        mx, my, mz = tool_measure()
        tx, ty = square_xy_cm(s)
        worst = max(worst, math.hypot(mx - tx, my - ty))
    p.resetJointState(robot, slider, 0.0); sync_extra_parts((0, 0, 0))
    print(f"IK check: {len(REACHABLE)}/64 squares reachable, worst fingertip error {worst * 10:.2f} mm")


print_reach_map()
self_test()
drive(cur)
wait(1.0)                                   # let the pieces settle on the board

if DIRECT:
    n_ok = 0
    for src, dst in DEMO_MOVES:
        n_ok += bool(pick_and_place(src, dst))
    print(f"\n{n_ok}/{len(DEMO_MOVES)} moves completed")
    p.disconnect()
    sys.exit(0)

if "--no-markers" not in sys.argv:
    dot = p.createVisualShape(p.GEOM_CYLINDER, radius=0.007, length=0.0006, rgbaColor=(0.1, 0.8, 0.2, 0.7))
    for s in REACHABLE:
        x, y = square_xy_cm(s)
        p.createMultiBody(0, -1, dot, [x / 100, y / 100, BOARD_TOP_CM / 100 + 0.0007])

p.resetDebugVisualizerCamera(0.95, 15, -38, [0, -0.08, 0.12])
ui_auto = p.addUserDebugParameter("Auto moves (1=play demo once)", 0, 1, 1)
ui_f1 = p.addUserDebugParameter("From file (0=a..7=h)", 0, 7, 4)
ui_r1 = p.addUserDebugParameter("From rank (1..8)", 1, 8, 2)
ui_f2 = p.addUserDebugParameter("To file (0=a..7=h)", 0, 7, 4)
ui_r2 = p.addUserDebugParameter("To rank (1..8)", 1, 8, 4)
ui_go = p.addUserDebugParameter("Move piece", 1, 0, 0)       # min > max makes it a button
ui_reset = p.addUserDebugParameter("Reset board", 1, 0, 0)
last_go, last_reset = p.readUserDebugParameter(ui_go), p.readUserDebugParameter(ui_reset)

sq = lambda f, r: "abcdefgh"[int(round(p.readUserDebugParameter(f)))] + str(int(round(p.readUserDebugParameter(r))))
played = False
while p.isConnected():
    if not played and p.readUserDebugParameter(ui_auto) > 0.5:
        for src, dst in DEMO_MOVES:
            if not p.isConnected():
                break
            pick_and_place(src, dst)
        played = True
    go, rs = p.readUserDebugParameter(ui_go), p.readUserDebugParameter(ui_reset)
    if rs != last_reset:
        last_reset = rs
        reset_board(); played = p.readUserDebugParameter(ui_auto) <= 0.5
    elif go != last_go:
        last_go = go
        pick_and_place(sq(ui_f1, ui_r1), sq(ui_f2, ui_r2))
    else:
        drive(cur)
        _tick()

if p.isConnected():
    p.disconnect()
