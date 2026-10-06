# Midnight

**A SCARA 4-DOF chess-playing robot.**

Midnight is a custom SCARA arm built to play chess — four degrees of freedom, designed for precise board reach and vertical piece placement. This repo holds the robot model, meshes, and tooling around it.

## What's here

| Path | Description |
|------|-------------|
| `Midnight/` | URDF model and preview |
| `meshes/` | Part meshes (STL / 3MF) |
| `fusion2urdf/` | Fusion 360 → URDF export helpers |
| `scripts/` | Viewing and mesh inspection utilities |
| `model/` | Additional model assets |

## Quick look

Preview the URDF:

```bash
python scripts/view_urdf.py
```

## Robot

- **Kinematics:** SCARA, 4 DOF  
- **Role:** Autonomous / assisted chess play  
- **CAD → sim:** Fusion 360 export via `fusion2urdf`

---

Built by [iamvaibhavmehra](https://github.com/iamvaibhavmehra).
