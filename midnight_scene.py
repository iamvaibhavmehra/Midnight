"""Builds the Midnight scene in PyBullet (robot, board, pieces, arm joints).

Importing this module connects to PyBullet (GUI, or DIRECT with --direct) and exposes
`robot`, `slider`, `JOINTS`, `sync_extra_parts(angles_deg)` and the board helpers.
"""
import math
import os
import struct
import sys
import time

import numpy as np
import pybullet as p
import pybullet_data

HERE = os.path.dirname(os.path.abspath(__file__))
DIRECT = "--direct" in sys.argv  # headless check: python hello_bullet.py --direct

p.connect(p.DIRECT if DIRECT else p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.loadURDF("plane.urdf")

# --- Fix-ups for the Fusion/ACDC4Robot export (original Midnight.urdf is untouched) ---
# The export is not a single kinematic tree, so we write a patched copy next to the
# meshes. All poses below were read from the Fusion design ("Midnight Full Version",
# world frame, cm -> m):
#   * MidnightBase-Wood: referenced as a parent link but never defined -> a 55x67.5x1.8 cm
#     oak slab with its origin at the world origin.
#   * SHF10D (+ Rod_Z2) and Chess-Board: no joint to the rest of the robot -> fixed
#     joints to the base at their Fusion placements.
WOOD_LINK = """<link name="MidnightBase-Wood">
        <visual><origin xyz="0 0 0.009" rpy="0 0 0"/>
            <geometry><box size="0.55 0.675 0.018"/></geometry></visual>
        <collision><origin xyz="0 0 0.009" rpy="0 0 0"/>
            <geometry><box size="0.55 0.675 0.018"/></geometry></collision>
    </link>"""
EXTRA_JOINTS = """
    <joint name="MidnightBase-Wood_Rigid-SHF10D" type="fixed">
        <origin xyz="0.000002 0.277502 0.034" rpy="0 0 3.141592653589793"/>
        <parent link="MidnightBase-Wood"/><child link="SHF10D"/>
    </joint>
    <joint name="MidnightBase-Wood_Rigid-Board" type="fixed">
        <origin xyz="0 -0.1125 0.0225" rpy="0 0 0"/>
        <parent link="MidnightBase-Wood"/><child link="Chess-Board"/>
    </joint>
</robot>"""
BOARD_ORIGIN = (0.0, -0.1125, 0.0225)

src = open(os.path.join(HERE, "Midnight.urdf"), encoding="utf-8").read()
if 'link name="MidnightBase-Wood"' not in src:
    src = src.replace('<link name="LowerBase">', WOOD_LINK + '\n    <link name="LowerBase">', 1)
    src = src.replace("</robot>", EXTRA_JOINTS)
urdf_path = os.path.join(HERE, "_Midnight_pybullet.urdf")
with open(urdf_path, "w", encoding="utf-8") as f:
    f.write(src)

robot = p.loadURDF(
    urdf_path,
    [0, 0, 0],
    p.getQuaternionFromEuler([0, 0, 0]),
    useFixedBase=True,
    flags=p.URDF_USE_INERTIA_FROM_FILE,
)

# --- Colours: STL/URDF carry none, so use the appearances from the Fusion design ---
COLORS = {
    "MidnightBase-Wood": (0.72, 0.52, 0.32),   # Oak - Glossy
    "LowerBase": (0.07, 0.07, 0.07),
    "SHF10D": (1, 1, 1), "SHF10E": (1, 1, 1), "SHF10F": (1, 1, 1),
    "Rod_Z1": (0.63, 0.63, 0.63), "Rod_Z2": (0.63, 0.63, 0.63), "Rod_Z3": (0.63, 0.63, 0.63),
    "Nema17Model": (0.96, 0.96, 0.96),
    "LeadScrew": (0.08, 0.08, 0.08),
    "LeadScrewNut": (1.0, 0.87, 0.55),
    "Arm1-Piece1": (0.07, 0.07, 0.07),
    "MidnightFusion_Arm1Piece2": (0.07, 0.07, 0.07),
    "Chess-Board": (0.965, 0.965, 0.953),
}
p.changeVisualShape(robot, -1, rgbaColor=(*COLORS["MidnightBase-Wood"], 1))
for i in range(p.getNumJoints(robot)):
    link = p.getJointInfo(robot, i)[12].decode()
    if link in COLORS:
        p.changeVisualShape(robot, i, rgbaColor=(*COLORS[link], 1))

# Chess board: the STL merges the cream slab and the 32 raised copper tiles. Find the
# tiles from the mesh (triangles on the top face) and overlay copper boxes on them.
def copper_tiles(stl_path, cell=0.0475):
    raw = open(stl_path, "rb").read()
    n = struct.unpack_from("<I", raw, 80)[0]
    tris = np.frombuffer(raw, dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]),
                         count=n, offset=84)["v"].reshape(-1, 3, 3) * 0.001
    ztop = tris[:, :, 2].max()
    top = tris[(tris[:, :, 2] > ztop - 1e-6).all(axis=1)]
    c = top.mean(axis=1)
    return sorted({(int(np.floor(x / cell)), int(np.floor(y / cell))) for x, y, _ in c}), ztop

cells, ztop = copper_tiles(os.path.join(HERE, "meshes", "Chess-Board.stl"))
cell = 0.0475
box = p.createVisualShape(p.GEOM_BOX, halfExtents=[cell / 2, cell / 2, 0.0006],
                          rgbaColor=(0.80, 0.51, 0.20, 1))
for i, j in cells:
    p.createMultiBody(0, -1, box, [BOARD_ORIGIN[0] + (i + 0.5) * cell,
                                   BOARD_ORIGIN[1] + (j + 0.5) * cell,
                                   BOARD_ORIGIN[2] + ztop - 0.0005 + 0.0001])

# --- Chess pieces ---
# No piece models exist in the CAD, so build simple turned-style pieces from primitives.
# Sizes in cm for a 4.75 cm square. Each primitive: (kind, params, (x, y, z) offset, optional rpy).
def _piece_shapes(kind):
    cyl = lambda r, h, z, x=0, y=0: ("cyl", (r, h), (x, y, z + h / 2))
    sph = lambda r, z, x=0, y=0: ("sph", (r,), (x, y, z))
    box = lambda sx, sy, sz, z, x=0, y=0, rpy=(0, 0, 0): ("box", (sx, sy, sz), (x, y, z + sz / 2), rpy)
    base = [cyl(1.75, 0.45, 0.0), cyl(1.45, 0.3, 0.45)]
    if kind == "P":
        return base + [cyl(0.95, 1.3, 0.75), cyl(1.25, 0.25, 2.0), sph(0.9, 2.65)]
    if kind == "R":
        return base + [cyl(1.25, 2.1, 0.75), cyl(1.55, 0.55, 2.85)] + [
            box(0.55, 0.55, 0.55, 3.4, x=dx, y=dy) for dx in (-1.0, 1.0) for dy in (-1.0, 1.0)]
    if kind == "N":
        return base + [cyl(1.1, 1.0, 0.75),
                       box(1.3, 1.5, 2.3, 1.55, y=0.1, rpy=(-0.25, 0, 0)),       # neck
                       box(1.1, 1.7, 0.95, 3.0, y=0.85, rpy=(0.35, 0, 0)),       # muzzle
                       box(0.35, 0.5, 0.6, 3.75, x=-0.4, y=-0.3), box(0.35, 0.5, 0.6, 3.75, x=0.4, y=-0.3)]
    if kind == "B":
        return base + [cyl(1.0, 2.0, 0.75), cyl(1.4, 0.25, 2.5), sph(1.0, 3.4), sph(0.35, 4.5)]
    if kind == "Q":
        return base + [cyl(1.2, 2.5, 0.75), cyl(1.7, 0.3, 3.0), cyl(1.4, 0.55, 3.3), sph(0.55, 4.15)]
    if kind == "K":
        return base + [cyl(1.25, 2.8, 0.75), cyl(1.7, 0.3, 3.3), cyl(1.4, 0.55, 3.6),
                       box(0.4, 0.4, 1.4, 4.1), box(1.1, 0.4, 0.4, 4.6)]
    raise ValueError(kind)

def make_piece(kind, rgba, x, y, z, yaw=0.0):
    vis = []
    for shp in _piece_shapes(kind):
        t, prm, off = shp[0], shp[1], [v / 100.0 for v in shp[2]]
        orn = p.getQuaternionFromEuler(shp[3]) if len(shp) > 3 else [0, 0, 0, 1]
        if t == "cyl":
            vis.append(p.createVisualShape(p.GEOM_CYLINDER, radius=prm[0] / 100, length=prm[1] / 100,
                                           rgbaColor=rgba, visualFramePosition=off, visualFrameOrientation=orn))
        elif t == "sph":
            vis.append(p.createVisualShape(p.GEOM_SPHERE, radius=prm[0] / 100, rgbaColor=rgba,
                                           visualFramePosition=off, visualFrameOrientation=orn))
        else:
            vis.append(p.createVisualShape(p.GEOM_BOX, halfExtents=[v / 200 for v in prm], rgbaColor=rgba,
                                           visualFramePosition=off, visualFrameOrientation=orn))
    n = len(vis) - 1
    return p.createMultiBody(
        0, -1, vis[0], [x, y, z], p.getQuaternionFromEuler([0, 0, yaw]),
        linkMasses=[0] * n, linkCollisionShapeIndices=[-1] * n, linkVisualShapeIndices=vis[1:],
        linkPositions=[[0, 0, 0]] * n, linkOrientations=[[0, 0, 0, 1]] * n,
        linkInertialFramePositions=[[0, 0, 0]] * n, linkInertialFrameOrientations=[[0, 0, 0, 1]] * n,
        linkParentIndices=[0] * n, linkJointTypes=[p.JOINT_FIXED] * n, linkJointAxis=[[0, 0, 1]] * n)

# White starts on the -y side (away from the arm), black on the +y (robot) side.
# a1 must be a dark (copper) square, so mirror the files if the tile pattern needs it.
files_mirrored = (-4, -4) not in set(cells)
WHITE, BLACK = (0.93, 0.89, 0.78, 1), (0.16, 0.11, 0.09, 1)
board_top = BOARD_ORIGIN[2] + ztop
back_rank = "RNBQKBNR"
for color, rgba, rank_back, rank_pawn, step, yaw in (
        ("white", WHITE, -4, -3, 1, 0.0), ("black", BLACK, 3, 2, -1, math.pi)):
    for f in range(8):
        i = (3 - f) if files_mirrored else (f - 4)        # file index -> board column
        x = BOARD_ORIGIN[0] + (i + 0.5) * cell
        for kind, j in ((back_rank[f], rank_back), ("P", rank_pawn)):
            make_piece(kind, rgba, x, BOARD_ORIGIN[1] + (j + 0.5) * cell, board_top, yaw)

# Find the joint that moves the arm up/down (the lead-screw nut slider).
slider = None
arm_link = None
for i in range(p.getNumJoints(robot)):
    info = p.getJointInfo(robot, i)
    print(i, info[1].decode(), info[2])
    if info[1].decode() == "LeadScrew_Slider-9":
        slider = i
        lo, hi = -0.25, 0.07   # Fusion "Slider 9" limits (-25..7 cm); the URDF export is stale (0.12)
    if info[12].decode() == "Arm1-Piece1":
        arm_link = i

# --- Arm sections added in Fusion after the URDF export (Arm2*, 120T_Stage2, Arm3*, 100T_Stage1) ---
# They are not in Midnight.urdf, so their meshes were exported from Fusion (meshes/*.stl, mm,
# in each component's own frame) and are placed with the occurrence transforms read from the
# Fusion design: (translation in cm, rotation about Z in degrees). They are rigidly attached to
# Arm1-Piece1 so they ride up and down with the arm.
EXTRA_ARM_PARTS = {
    "100T_Stage1Final": ((0.0232, 2.0023, 40.06), 0.0),
    "Arm2Top":          ((0.0002, 21.2502, 4.7879), -90.0),
    "Arm2Base":         ((0.0002, 21.2502, 4.7879), -90.0),
    "Arm2Base2":        ((0.0002, 21.2502, 4.7879), -90.0),
    "120T_Stage2":      ((-35.6722, -3.76, 4.9879), -20.0),
    "Arm3Lid":          ((0.0002, 21.2502, -15.9121), -90.0),
    "Arm3":             ((0.0002, 21.2502, 4.0879), -90.0),
    "Nema17Stage1":     ((0.0232, 16.4773, 33.82), 90.0),
    "60T_Stage3":       ((0.0232, -24.6277, 27.78), -130.0),
    "Arm1Connector":    ((-14.7363, -12.273, -6.8421), -40.0),
    # Gripper on the end of Arm 3: (translation cm, yaw deg, roll deg about X)
    "GripperConnector": ((-0.2335, -24.4123, 35.6), -40.0),
    "GripperHousing":   ((0.2385, -24.371, 18.72), -130.0, 180.0),
    "RackA":            ((2.2329, -24.3084, 18.01), -130.0, 90.0),   # finger, slides along +axis
    "RackB":            ((-2.2249, -24.9148, 18.01), 50.0, 90.0),    # finger, slides along -axis
}
# Finger sliders (Fusion "Slider 41/42", 0 .. -1.5 cm, both toward the gripper centre):
# direction in the world at the Fusion pose that each rack moves when the gripper CLOSES.
RACK_CLOSE_DIR = {"RackA": (-0.6428, -0.7660), "RackB": (0.6428, 0.7660)}
GRIP_MAX_CM = 1.5
# Parts added to the arm in Fusion that have several bodies: one STL per body (colour from Fusion).
EXTRA_COLORS = {"Nema17Stage1": (245, 245, 246)}     # everything else defaults to black
_MOTOR_BODY_COLORS = [(160, 160, 160), (245, 245, 246), (203, 203, 203), (25, 25, 25), (245, 245, 246)]
for _stem, _pose in (("NEMA17_stepper_motor_v39", ((0.0232, 2.0023, 31.14), 0.0)),
                     ("NEMA_17_x_23mm_Stepper_Motor", ((0.0232, -16.7277, 26.41), 0.0))):
    for _i, _c in enumerate(_MOTOR_BODY_COLORS):
        EXTRA_ARM_PARTS[f"{_stem}__{_i}"] = _pose
        EXTRA_COLORS[f"{_stem}__{_i}"] = _c
MOTOR_J1 = tuple(f"NEMA17_stepper_motor_v39__{i}" for i in range(5))        # on Arm2Base
MOTOR_J2 = tuple(f"NEMA_17_x_23mm_Stepper_Motor__{i}" for i in range(5))    # on Arm3

# Fixed top frame (rods' upper bracket + caps): static, so no joint needed.
STATIC_PARTS = {
    "Upper_Base": ((0.0005, 21.2501, 2.7), 30.0, (0, 0, 0)),
    "SHF10":      ((5.6294, 18.0002, 52.0), 60.0, (255, 255, 255)),
    "SHF10B":     ((0.0002, 27.7502, 52.0), 180.0, (255, 255, 255)),
    "SHF10C":     ((-5.629, 18.0002, 52.0), -60.0, (255, 255, 255)),
}
def _rgba(c):
    return (0.07, 0.07, 0.07, 1) if c == (0, 0, 0) else (c[0] / 255, c[1] / 255, c[2] / 255, 1)
for _name, (_t, _yaw, _c) in STATIC_PARTS.items():
    _vis = p.createVisualShape(p.GEOM_MESH, fileName=os.path.join(HERE, "meshes", _name + ".stl"),
                               meshScale=[0.001] * 3, rgbaColor=_rgba(_c))
    p.createMultiBody(0, -1, _vis, [v / 100.0 for v in _t],
                      p.getQuaternionFromEuler([0, 0, math.radians(_yaw)]))
FUSION_ARM1_ZMIN, FUSION_ARM1_ZMAX = 0.3754, 0.4214   # Arm1 Piece1 z-range in the Fusion pose

# Joint value at which the URDF arm sits where it does in Fusion (the AABB padding cancels out).
p.resetJointState(robot, slider, 0.0)
aabb = p.getAABB(robot, arm_link)
pad = ((aabb[1][2] - aabb[0][2]) - (FUSION_ARM1_ZMAX - FUSION_ARM1_ZMIN)) / 2
q_ref = (FUSION_ARM1_ZMIN - pad) - aabb[0][2]
p.resetJointState(robot, slider, q_ref)
print(f"Fusion pose corresponds to slider q={q_ref:+.4f} m")

def _arm_frame():
    st = p.getLinkState(robot, arm_link, computeForwardKinematics=True)
    return st[4], st[5]

_ARM_REF = _arm_frame()
_ARM_REF_INV = p.invertTransform(*_ARM_REF)
_ID = ([0, 0, 0], [0, 0, 0, 1])

# The three revolute joints from Fusion, in chain order. Pivots are vertical axes through the
# given world point (cm -> m) at the Fusion pose. Fusion rotates about -Z, so its limits
# [min, max] become [-max, -min] about +Z. "parts" move with this joint and every joint before it.
JOINTS = [
    dict(name="Joint 1", pivot=(0.0232, 2.0023, 39.91), fusion_limits=(-120.0, 135.0),
         parts=("100T_Stage1Final", "Arm2Top", "Arm2Base", "Arm2Base2") + MOTOR_J1, demo=(20.0, 80.0, 8.0)),
    dict(name="Joint 2", pivot=(0.0232, -16.7277, 32.71), fusion_limits=(-210.0, 110.0),
         parts=("120T_Stage2", "Arm3Lid", "Arm3") + MOTOR_J2, demo=(60.0, 100.0, 5.0)),
    dict(name="Joint 3", pivot=(0.0232, -24.6277, 27.28), fusion_limits=(-30.0, 35.0),
         parts=("60T_Stage3", "Arm1Connector", "GripperConnector", "GripperHousing", "RackA", "RackB"), demo=(-2.5, 25.0, 3.5)),
]
for jt in JOINTS:
    jt["pivot"] = tuple(v / 100.0 for v in jt["pivot"])
    jt["limits"] = (-jt["fusion_limits"][1], -jt["fusion_limits"][0])   # about +Z, degrees
    jt["level"] = {n: i + 1 for i, n in enumerate(jt["parts"])}

extra_parts = []   # (name, bodyId, world pose at the Fusion pose)
for name, spec in EXTRA_ARM_PARTS.items():
    t_cm, yaw_deg = spec[0], spec[1]
    roll_deg = spec[2] if len(spec) > 2 else 0.0
    pos = [v / 100.0 for v in t_cm]
    orn = p.multiplyTransforms([0, 0, 0], p.getQuaternionFromEuler([0, 0, math.radians(yaw_deg)]),
                               [0, 0, 0], p.getQuaternionFromEuler([math.radians(roll_deg), 0, 0]))[1]
    vis = p.createVisualShape(p.GEOM_MESH, fileName=os.path.join(HERE, "meshes", name + ".stl"),
                              meshScale=[0.001] * 3, rgbaColor=_rgba(EXTRA_COLORS.get(name, (0, 0, 0))))
    extra_parts.append((name, p.createMultiBody(0, -1, vis, pos, orn), (pos, orn)))

_PART_LEVEL = {n: i for i, jt in enumerate(JOINTS) for n in jt["parts"]}   # joint index per part

def _about_pivot(pivot, angle):
    """Pose of a rotation by `angle` (rad, about +Z) around a vertical axis through `pivot`."""
    q = p.getQuaternionFromEuler([0, 0, angle])
    rp = p.multiplyTransforms([0, 0, 0], q, pivot, [0, 0, 0, 1])[0]
    return [pivot[i] - rp[i] for i in range(3)], q

def sync_extra_parts(angles_deg=(0.0, 0.0, 0.0), grip_cm=0.0):
    """Place the added arm sections: ride on Arm1-Piece1 (slider), then apply the revolutes in turn.
    `grip_cm` closes the two gripper fingers (0 = open .. 1.5 = closed)."""
    grip = min(max(grip_cm, 0.0), GRIP_MAX_CM) / 100.0
    ap, ao = _arm_frame()
    chain = [p.multiplyTransforms(ap, ao, *_ARM_REF_INV)]       # arm motion since the Fusion pose
    for jt, ang in zip(JOINTS, angles_deg):
        lo_d, hi_d = jt["limits"]
        rad = math.radians(min(max(ang, lo_d), hi_d))
        chain.append(p.multiplyTransforms(*chain[-1], *_about_pivot(jt["pivot"], rad)))
    for name, body, (wp, wo) in extra_parts:
        if name in RACK_CLOSE_DIR:
            dx, dy = RACK_CLOSE_DIR[name]
            wp = [wp[0] + dx * grip, wp[1] + dy * grip, wp[2]]
        pos, orn = p.multiplyTransforms(*chain[_PART_LEVEL.get(name, -1) + 1], wp, wo)
        p.resetBasePositionAndOrientation(body, pos, orn)

