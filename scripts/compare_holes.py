"""Compare base/arm hole layout vs current rod axes."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

mesh_dir = Path(r"C:\Users\vaibh\Midnight\Midnight\meshes")


def find_circles(verts2d, expected_r=5.0, tol=1.0, min_ring=8):
    centers = []
    xs = np.linspace(verts2d[:, 0].min() - 5, verts2d[:, 0].max() + 5, 70)
    ys = np.linspace(verts2d[:, 1].min() - 5, verts2d[:, 1].max() + 5, 70)
    for x in xs:
        for y in ys:
            d = np.sqrt((verts2d[:, 0] - x) ** 2 + (verts2d[:, 1] - y) ** 2)
            on_ring = int(np.sum(np.abs(d - expected_r) < tol))
            inside = int(np.sum(d < expected_r - tol))
            if on_ring >= min_ring and inside <= 2:
                centers.append((float(x), float(y), on_ring))
    centers = sorted(centers, key=lambda c: -c[2])
    kept = []
    for c in centers:
        if all(np.hypot(c[0] - k[0], c[1] - k[1]) > 5 for k in kept):
            kept.append(c)
        if len(kept) >= 12:
            break
    return kept


def cluster_report(label, kept):
    print(f"\n{label}:")
    for a, b, n in kept:
        print(f"  ({a:7.2f}, {b:7.2f}) mm  ring_pts={n}  r_from_origin={np.hypot(a,b):.2f}")


def main() -> None:
    base = trimesh.load(str(mesh_dir / "base_link.stl"), force="mesh")
    arm = trimesh.load(str(mesh_dir / "Arm1-Piece1.stl"), force="mesh")
    bv, av = np.asarray(base.vertices), np.asarray(arm.vertices)

    print("BASE bounds", base.bounds)
    # Top face of base ~ z=7 to 14
    for z0 in [0, 5, 7, 10, 14]:
        band = bv[np.abs(bv[:, 2] - z0) < 1.0]
        if len(band) < 15:
            continue
        for r in (5.0, 5.5, 6.0, 8.0, 10.0):
            kept = find_circles(band[:, :2], expected_r=r, tol=1.2, min_ring=6)
            if kept:
                cluster_report(f"BASE z={z0} r={r}", kept)

    print("\n" + "=" * 60)
    print("ARM bottom face holes (z~318)")
    band = av[np.abs(av[:, 2] - 318) < 2]
    for r in (5.0, 5.5, 6.0):
        kept = find_circles(band[:, :2], expected_r=r, tol=1.2, min_ring=8)
        cluster_report(f"ARM z=318 r={r}", kept)

    print("\n" + "=" * 60)
    print("Current ROD axes (from earlier analysis):")
    rods = [(22.50, -38.97), (22.50, 38.97), (-45.00, 0.00), (0.0, 0.0)]  # last is leadscrew
    names = ["Rod_Z_1", "Rod_Z_2", "Rod_Z_3", "LeadScrew"]
    for n, (x, y) in zip(names, rods):
        print(f"  {n:12s} ({x:7.2f}, {y:7.2f})")

    print("\nSHF10 body-frame joints on base:")
    shfs = [(46.35, -48.29), (46.35, 48.29), (-65.00, -16.00)]
    for i, (x, y) in enumerate(shfs, 1):
        print(f"  SHF10_{i}     ({x:7.2f}, {y:7.2f})")


if __name__ == "__main__":
    main()
