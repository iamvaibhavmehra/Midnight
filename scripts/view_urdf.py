"""Load Midnight URDF in MuJoCo viewer and save a preview PNG."""
from __future__ import annotations

import struct
import zlib
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

URDF = Path(r"C:\Users\vaibh\Midnight\Midnight\Midnight.urdf")
PREVIEW = Path(r"C:\Users\vaibh\Midnight\Midnight\preview.png")


def write_png(path: Path, rgb: np.ndarray) -> None:
    """Write an HxWx3 uint8 array as PNG (stdlib only)."""
    h, w, c = rgb.shape
    assert c == 3
    raw = b"".join(b"\x00" + rgb[i].tobytes() for i in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
    path.write_bytes(png)


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(URDF))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    print(f"Loaded OK: nbody={model.nbody} nq={model.nq} ngeom={model.ngeom}")
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        print(f"  body {i}: {name}  xpos={data.xpos[i]}")

    renderer = mujoco.Renderer(model, 480, 640)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.lookat[:] = [0.0, 0.0, 0.15]
    cam.distance = 1.2
    cam.azimuth = 140
    cam.elevation = -25
    renderer.update_scene(data, camera=cam)
    write_png(PREVIEW, renderer.render())
    print(f"Preview saved: {PREVIEW}")

    print("Opening MuJoCo viewer — close the window when done.")
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
