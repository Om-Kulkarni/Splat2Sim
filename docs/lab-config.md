# Lab config

Per-lab YAML describing the board, robot, cameras and cubes of a workcell. Used by `anchor`, `collision`, `report` and
`env` ([Architecture](architecture.md)).

`labs/<lab>.yaml`, schema in `splat2sim/config.py` (units: m, kg, rad for joints, degrees for rpy/fov; a pose is
`xyz` + `rpy` with `R = Rz*Ry*Rx`, z-up, world = FR3 base).

| Section | Content | Used by |
|---|---|---|
| `charuco` | board definition (incl. `square_length_m`) | [Anchor](stages/anchor.md) |
| `board_to_base` | measured board pose in the robot base frame | [Anchor](stages/anchor.md) |
| `robot` | scan pose (`scan_pose_joints`), gripper, `coupling_offset` | [Report](stages/report.md), [Task environment](task-environment.md) |
| `policy_cameras` | `wrist`, `front` | [Task environment](task-environment.md) |
| `cubes` | cube spec | [Task environment](task-environment.md) |
| `overrides` (optional) | `table_box`, `pedestal_box` | [Collision](stages/collision.md) |
| `collision` (optional) | clearance defaults | [Collision](stages/collision.md) |
| `env` (optional) | `episode_length_s`, `reach_min_m`, `reach_max_m`, `spawn_margin_m` | [Task environment](task-environment.md) |

A per-run file (`--override file.yaml`) is deep-merged on top (dicts merge, lists replace).

## Add a lab

Copy `labs/robolab.yaml` to `labs/<name>.yaml`, measure the board square length and `board_to_base`, then
`./r2s build <capture> --lab <name>`. Values marked TODO in `robolab.yaml` are placeholders.
