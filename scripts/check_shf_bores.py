"""Check SHF10 bore axes vs rod axes in world frame."""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import trimesh

mesh_dir = Path(r"C:\Users\vaibh\Midnight\Midnight\meshes")
urdf = Path(r"C:\Users\vaibh\Midnight\Midnight\Midnight_fixed.urdf")


def find_circles(verts2d, expected_r=5.0, tol=1.0, min_ring=6):
    centers = []
    pad = 5
    xs = np.linspace(verts2d[:, 0].min() - pad, verts2d[:, 0].max() + pad, 50)
    ys = np.linspace(verts2d[:, 1].min() - pad, verts2d[:, 1].max() + pad, 50)
    for x in xs:
        for y in ys:
            d = np.sqrt((verts2d[:, 0] - x) ** 2 + (verts2d[:, 1] - y) ** 2)
            on_ring = int(np.sum(np.abs(d - expected_r) < tol))
            inside = int(np.sum(d < expected_r - tol))
            if on_ring >= min_ring and inside <= 1:
                centers.append((float(x), float(y), on_ring))
    centers = sorted(centers, key=lambda c: -c[2])
    kept = []
    for c in centers:
        if all(np.hypot(c[0] - k[0], c[1] - k[1]) > 3 for k in kept):
            kept.append(c)
        if len(kept) >= 5:
            break
    return kept


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(urdf))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    print("World geom positions (mm):")
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
        if not any(k in name for k in ("SHF", "Rod")):
            continue
        p = data.geom_xpos[i] * 1000
        print(f"  {name:28s} ({p[0]:7.2f}, {p[1]:7.2f}, {p[2]:7.2f})")

        # Transform mesh verts to world and find 10mm bore (circle in plane normal to rod = Z)
        mi = model.geom_dataid[i]
        if mi < 0:
            continue
        vadr = model.mesh_vertadr[mi]
        vnum = model.mesh_vertnum[mi]
        local = model.mesh_vert[vadr : vadr + vnum]
        R = data.geom_xmat[i].reshape(3, 3)
        t = data.geom_xpos[i]
        world = (R @ local.T).T + t

        if "SHF" in name:
            # SHF10 bore is vertical (along world Z); search XY circle at mid Z of part
            zmid = world[:, 2].mean()
            band = world[np.abs(world[:, 2] - zmid) < 0.003]
            if len(band) > 10:
                kept = find_circles(band[:, :2] * 1000, expected_r=5.0, tol=1.5)
                print(f"    bore candidates at z~{zmid*1000:.1f}: {[(round(a,2),round(b,2),n) for a,b,n in kept]}")

    # Raw SHF10 STL hole in local frame
    shf = trimesh.load(str(mesh_dir / "SHF10.stl"), force="mesh")
    v = np.asarray(shf.vertices)
    print("\nRaw SHF10 STL bounds (mm):", shf.bounds)
    # Bore along Y in this part? extents Y=0..33. Or along Z?
    # Check circles in XZ (bore along Y) and XY (bore along Z)
    for axis, label, idx in [("Y", "XZ plane (bore along Y)", [0, 2]), ("Z", "XY plane (bore along Z)", [0, 1])]:
        mid = v[:, {"Y": 1, "Z": 2}[axis]].mean()
        band = v[np.abs(v[:, {"Y": 1, "Z": 2}[axis]] - mid) < 1.0]
        kept = find_circles(band[:, idx], expected_r=5.0, tol=1.2)
        print(f"  Local {label}: {[(round(a,2),round(b,2),n) for a,b,n in kept]}")


if __name__ == "__main__":
    main()
