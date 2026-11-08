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

# Find the joint that moves the arm up/down (the lead-screw nut slider).
slider = None
arm_link = None
for i in range(p.getNumJoints(robot)):
    info = p.getJointInfo(robot, i)
    print(i, info[1].decode(), info[2])
    if info[1].decode() == "LeadScrew_Slider-9":
        slider = i
        lo, hi = info[8], info[9]
    if info[12].decode() == "Arm1-Piece1":
        arm_link = i

# --- Arm sections added in Fusion after the URDF export (Arm2*, 120T_Stage2, Arm3*, 100T_Stage1) ---
# They are not in Midnight.urdf, so their meshes were exported from Fusion (meshes/*.stl, mm,
# in each component's own frame) and are placed with the occurrence transforms read from the
# Fusion design: (translation in cm, rotation about Z in degrees). They are rigidly attached to
# Arm1-Piece1 so they ride up and down with the arm.
EXTRA_ARM_PARTS = {
    "100T_Stage1Final": ((0.0232, 2.0023, 40.06), -140.0),
    "Arm2Top":          ((0.0002, 21.2502, 4.7879), -90.0),
    "Arm2Base":         ((0.0002, 21.2502, 4.7879), -90.0),
    "Arm2Base2":        ((0.0002, 21.2502, 4.7879), -90.0),
    "120T_Stage2":      ((24.4173, 12.3798, 4.9879), -130.0),
    "Arm3Lid":          ((0.0002, 21.2502, -15.9121), -90.0),
    "Arm3":             ((0.0002, 21.2502, 4.0879), -90.0),
}
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

# Revolute joints from Fusion (both vertical axes; Fusion range -320..0 deg about -Z
# == 0..+320 deg about +Z). Pivots in the world frame at the Fusion pose, cm -> m.
PIVOT1 = (0.0232 / 100, 2.0023 / 100, 0.3991)    # "Revolute 13": 100T_Stage1Final on Arm1 Piece2
PIVOT2 = (0.0232 / 100, -16.7277 / 100, 0.3271)  # "Revolute 17": 120T_Stage2 on Arm2Base2
JOINT_LIMIT_DEG = (0.0, 320.0)
GROUP1 = ("100T_Stage1Final", "Arm2Top", "Arm2Base", "Arm2Base2")   # rotate with joint 1
GROUP2 = ("120T_Stage2", "Arm3Lid", "Arm3")                         # also rotate with joint 2

extra_parts = []   # (name, bodyId, world pose at the Fusion pose)
for name, (t_cm, yaw_deg) in EXTRA_ARM_PARTS.items():
    pos = [v / 100.0 for v in t_cm]
    orn = p.getQuaternionFromEuler([0, 0, math.radians(yaw_deg)])
    vis = p.createVisualShape(p.GEOM_MESH, fileName=os.path.join(HERE, "meshes", name + ".stl"),
                              meshScale=[0.001] * 3, rgbaColor=(0.07, 0.07, 0.07, 1))
    extra_parts.append((name, p.createMultiBody(0, -1, vis, pos, orn), (pos, orn)))

def _about_pivot(pivot, angle):
    """Pose of a rotation by `angle` (rad, about +Z) around a vertical axis through `pivot`."""
    q = p.getQuaternionFromEuler([0, 0, angle])
    rp = p.multiplyTransforms([0, 0, 0], q, pivot, [0, 0, 0, 1])[0]
    return [pivot[i] - rp[i] for i in range(3)], q

def sync_extra_parts(j1_deg=0.0, j2_deg=0.0):
    """Place the added arm sections: ride on Arm1-Piece1 (slider) and apply both revolutes."""
    lo_d, hi_d = JOINT_LIMIT_DEG
    j1 = math.radians(min(max(j1_deg, lo_d), hi_d))
    j2 = math.radians(min(max(j2_deg, lo_d), hi_d))
    ap, ao = _arm_frame()
    d = p.multiplyTransforms(ap, ao, *_ARM_REF_INV)            # arm motion since the Fusion pose
    t1 = p.multiplyTransforms(*d, *_about_pivot(PIVOT1, j1))   # + joint 1
    t2 = p.multiplyTransforms(*t1, *_about_pivot(PIVOT2, j2))  # + joint 2 (child of joint 1)
    for name, body, (wp, wo) in extra_parts:
        base = t2 if name in GROUP2 else t1 if name in GROUP1 else d
        pos, orn = p.multiplyTransforms(*base, wp, wo)
        p.resetBasePositionAndOrientation(body, pos, orn)

# Auto demo: all three joints oscillate. In the GUI you can switch it off with the
# "Auto demo" slider and drive the joints by hand with the other sliders.
mid, amp = (lo + hi) / 2, (hi - lo) / 2
J1_MAX, J2_MAX = 110.0, 150.0     # demo sweep, degrees (hardware limit is 320)

if not DIRECT:
    p.resetDebugVisualizerCamera(1.1, 40, -25, [0, -0.05, 0.25])
    ui_auto = p.addUserDebugParameter("Auto demo (1=on, 0=manual)", 0, 1, 1)
    ui_z = p.addUserDebugParameter("Z slider (m)", lo, hi, 0.0)
    ui_j1 = p.addUserDebugParameter("Joint 1 (deg)", *JOINT_LIMIT_DEG, 0.0)
    ui_j2 = p.addUserDebugParameter("Joint 2 (deg)", *JOINT_LIMIT_DEG, 0.0)

steps = 240 * 16 if DIRECT else None
t = 0.0
dt = 1.0 / 240.0
n = 0
while p.isConnected() and (steps is None or n < steps):
    if DIRECT or p.readUserDebugParameter(ui_auto) > 0.5:
        target = mid + amp * math.sin(2 * math.pi * t / 6.0)
        j1 = J1_MAX * (1 - math.cos(2 * math.pi * t / 8.0)) / 2
        j2 = J2_MAX * (1 - math.cos(2 * math.pi * t / 5.0)) / 2
    else:
        target = p.readUserDebugParameter(ui_z)
        j1 = p.readUserDebugParameter(ui_j1)
        j2 = p.readUserDebugParameter(ui_j2)
    p.setJointMotorControl2(
        robot, slider, p.POSITION_CONTROL,
        targetPosition=target, force=1000, maxVelocity=0.2,
    )
    p.stepSimulation()
    sync_extra_parts(j1, j2)
    if n % 240 == 0:
        print(f"t={t:5.2f} z={p.getJointState(robot, slider)[0]:+.3f} m  j1={j1:6.1f} deg  j2={j2:6.1f} deg")
    if not DIRECT:
        time.sleep(dt)
    t += dt
    n += 1

if p.isConnected():
    p.disconnect()
