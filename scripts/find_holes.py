"""Find hole centers in Arm mesh and compare to rod positions."""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import trimesh

mesh_dir = Path(r"C:\Users\vaibh\Midnight\Midnight\meshes")
urdf = Path(r"C:\Users\vaibh\Midnight\Midnight\Midnight_fixed.urdf")


def find_circles(verts2d: np.ndarray, expected_r: float = 5.0, tol: float = 1.0):
    centers = []
    xs = np.linspace(verts2d[:, 0].min(), verts2d[:, 0].max(), 60)
    ys = np.linspace(verts2d[:, 1].min(), verts2d[:, 1].max(), 60)
    for x in xs:
        for y in ys:
            d = np.sqrt((verts2d[:, 0] - x) ** 2 + (verts2d[:, 1] - y) ** 2)
            on_ring = int(np.sum(np.abs(d - expected_r) < tol))
            inside = int(np.sum(d < expected_r - tol))
            if on_ring >= 8 and inside <= 2:
                centers.append((float(x), float(y), on_ring))
    centers = sorted(centers, key=lambda c: -c[2])
    kept = []
    for c in centers:
        if all(np.hypot(c[0] - k[0], c[1] - k[1]) > 4 for k in kept):
            kept.append(c)
        if len(kept) >= 10:
            break
    return kept


def main() -> None:
    arm = trimesh.load(str(mesh_dir / "Arm1-Piece1.stl"), force="mesh")
    verts = np.asarray(arm.vertices)
    print("Arm bounds (mm, assembly frame):", arm.bounds)

    for z0 in [318, 330, 340, 350, 360]:
        band = verts[np.abs(verts[:, 2] - z0) < 1.5]
        if len(band) < 20:
            print(f"Z={z0}: few verts ({len(band)})")
            continue
        for r in (4.0, 5.0, 5.5, 6.0, 8.0):
            kept = find_circles(band[:, :2], expected_r=r, tol=1.2)
            if kept:
                print(f"Z={z0} r={r}: {[(round(a, 2), round(b, 2), n) for a, b, n in kept]}")

    # World-frame rod / SHF / arm positions from MuJoCo
    model = mujoco.MjModel.from_xml_path(str(urdf))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    print("\n=== MuJoCo world geom XY (m) ===")
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
        if any(k in name for k in ("Rod", "SHF", "LeadScrew_c", "Arm", "base")):
            p = data.geom_xpos[i]
            print(f"{name:30s} xy=({p[0]*1000:7.2f},{p[1]*1000:7.2f}) mm  z={p[2]*1000:7.2f}")

    # Transform arm STL verts to world using geom pose
    # Find arm geom
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
        if "Arm1" not in name:
            continue
        # MuJoCo stores processed mesh verts in geom-local frame already
        # Use mesh 6 verts + geom pose
        mi = model.geom_dataid[i]
        vadr = model.mesh_vertadr[mi]
        vnum = model.mesh_vertnum[mi]
        local = model.mesh_vert[vadr : vadr + vnum]
        R = data.geom_xmat[i].reshape(3, 3)
        t = data.geom_xpos[i]
        world = (R @ local.T).T + t
        print("\nArm mesh in world (m):")
        print("  bounds", world.min(0), world.max(0))

        # Search holes in world XY at arm mid Z
        zmid = world[:, 2].mean()
        band = world[np.abs(world[:, 2] - zmid) < 0.005]
        print(f"  band at z={zmid:.4f} n={len(band)}")
        # holes ~5mm radius = 0.005m
        for r in (0.004, 0.005, 0.0055, 0.008):
            kept = find_circles(band[:, :2] * 1000, expected_r=r * 1000, tol=1.5)  # work in mm
            if kept:
                print(f"  world holes r={r*1000:.1f}mm: {[(round(a,2), round(b,2), n) for a,b,n in kept]}")

        # Also dump XY of all band points roughly
        if len(band) > 0:
            # Find concave/hole by looking at radial empty regions from lead screw axis (0,0)
            print("  band XY mm sample unique-ish:")
            xy = np.round(band[:, :2] * 1000, 1)
            # histogram of radius
            rad = np.sqrt(xy[:, 0] ** 2 + xy[:, 1] ** 2)
            print(f"  radii mm: min={rad.min():.1f} max={rad.max():.1f}")


if __name__ == "__main__":
    main()
