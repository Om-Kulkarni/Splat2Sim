# Franka + Robotiq 2F-85 asset for Isaac Sim 6 / Isaac Lab

Researched 2026-10-17.
Note: the web-fetch summaries were produced by a small model; claims marked (UNVERIFIED-IN-SIM) were not run on hardware.

## TL;DR

- **Recommended: use NVIDIA's bundled `Robots/FrankaRobotics/FrankaPanda/franka.usd` with variant `Gripper=Robotiq_2F_85`.** Isaac Lab already ships `FRANKA_ROBOTIQ_GRIPPER_CFG` for exactly this (arm + 2F-85, mimic joint, passive-joint actuators). No URDF import needed.
- That bundled asset is a **Panda** (joints `panda_joint1..7`). **No FR3 + Robotiq variant is listed** in Isaac Sim's asset catalog; FR3 (`FrankaFR3/fr3.usd`) is listed as not validated. If the lab arm is an FR3, the bundled asset is kinematically close but not exact (different joint limits, see below) and you must build an FR3+2F-85 asset yourself (composition recipe below).
- The 2F-85 mechanism is a closed-loop linkage. Isaac Sim handles it by excluding the inner-knuckle joints from the articulation (breaking the loop) and using a mimic joint on `right_outer_knuckle_joint` driven by `finger_joint` (gearing -1). Control surface = one joint, `finger_joint`.
- Robotiq has no first-party sim assets; community assets have contact bugs. Validate asymmetric/edge grasps on cubes.
- Isaac Lab targeting Isaac Sim 6.x is the `develop` / `release/3.0.0` branch (Isaac Sim 6.1); `main` is Isaac Sim 4.5-5.1.

## 1. Panda vs FR3: how to tell them apart

Franka's own docs ([robot specs](https://frankarobotics.github.io/docs/robot_specifications.html)):

| | Panda (FER) | FR3 |
|---|---|---|
| q2 limit | +-1.7628 | +-1.8361 |
| q1/q3 limit | +-2.8973 | +-2.9007 |
| q4 range | [-3.0718, -0.0698] | [-3.0770, -0.1169] |
| q5 / q7 limit | +-2.8973 | +-2.8763 / +-3.0508 |
| q6 range | [-0.0175, 3.7525] | [0.4398, 4.6216] |
| q1-4 max vel | 2.175 rad/s | 2.62 rad/s |

Practical checks (in order of ease):
1. **Desk (web UI) Settings > Dashboard shows arm type**: `Arm3P`=fp3, `Arm3R`=fr3, `Arm3Rv2`=fr3v2, `Arm3Rv2.1`=fr3v2_1; may display wrongly on Panda (FER) or system images older than 5.9.1 ([franka_description README](https://github.com/frankarobotics/franka_description)). fr3/fr3v2/fr3v2_1 are hard to tell apart.
2. Read joint limits / state via libfranka or franka_ros2 (e.g. `q6` can never be negative on FR3; q6 min 0.44 vs -0.0175).
3. Nameplate/labels on the arm (the Franka docs fetched do not describe physical markings; read the sticker/serial and the Franka docs). Franka states `fer` is "not supported" in franka_description; use fr3 or fp3.
4. Note the Isaac Lab home pose used in the config (q6 = 3.037) is inside both ranges; q6 of 4.x only exists on FR3.

## 2. Candidate sources

| Source | What | Format | License | Verdict |
|---|---|---|---|---|
| [Isaac Sim robot assets (manipulators)](https://docs.isaacsim.omniverse.nvidia.com/latest/assets/usd_assets_robots_manipulator.html) `FrankaRobotics/FrankaPanda/franka.usd` | Panda; accessories: AlternateFinger, Default, **Robotiq_2F_85**; 10 joints, 9 DOF; listed Validated (Multiphysics path `FrankaRobotics/FrankaPanda/franka/franka.usda`) | USD (+ multiphysics usda) | Apache-2.0 (Franka) | **Use** (Panda) |
| same page, `FrankaRobotics/FrankaFR3/fr3.usd` | FR3 arm only, 9 DOF, not validated | USD | Apache-2.0 | Base for FR3 build |
| same page, `Robotiq/2F-85/Robotiq_2F_85_edit.usd` | Standalone 2F-85, 8 joints/6 DOF, Validated multiphysics `Robotiq/2F-85/Robotiq_2F_85.usda` | USD | "3D Content Sharing Agreement" (Robotiq) | Gripper source for FR3 compose |
| [Isaac Lab `franka.py`](https://github.com/isaac-sim/IsaacLab/blob/develop/source/isaaclab_assets/isaaclab_assets/robots/franka.py) | `FRANKA_ROBOTIQ_GRIPPER_CFG` ArticulationCfg | Python | BSD-3 (robotiq files under ROS-Industrial BSD, [license](https://github.com/isaac-sim/IsaacLab/blob/main/docs/licenses/assets/robotiq-license.txt)) | **Use** |
| [franka_description](https://github.com/frankarobotics/franka_description) | Official FR3/FP3/FER URDF generator (`./scripts/create_urdf.sh fr3`; `--robot-ee franka_hand\|cobot_pump`, `--no-ee`) | URDF | Apache-2.0 | FR3 kinematics/meshes if you need own URDF; no Robotiq |
| [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) | `franka_emika_panda`, `franka_fr3`, `franka_fr3_v2`, `robotiq_2f85`, `robotiq_2f85_v4` | MJCF | Apache-2.0 / BSD-2 (Robotiq) | Reference for gripper dynamics; needs MJCF->USD |
| [UWLab](https://github.com/UW-Lab/UWLab) | Isaac Lab 3.0 EA / Isaac Sim 6.1 framework (BSD-3); hosts a calibrated 2F-85 USD in [HF dataset](https://huggingface.co/datasets/UW-Lab/uwlab-assets/blob/main/Robots/UniversalRobots/2f85RobotiqGripperCalibrated/robotiq_2f85_gripper_calibrated.usd) | USD | not stated for the asset (check) | Fallback if NVIDIA gripper misbehaves; ROS-I rated best |

ROS-Industrial's comparison ([From URDF to SimReady](https://rosindustrial.org/news/2026/7/23/from-urdf-to-simready-what-a-robotiq-gripper-taught-us-about-simulation-assets)): Robotiq provides no first-party sim assets; Isaac Sim's bundled Robotiq asset was hard to restructure (articulation root), not instanceable, and **broke when a mimic joint hit an object asymmetrically at one pad**; UW-Lab calibrated USD handled that case best; none reproduced real pad angling near the pad base. Recommends: keep the coupled mimic abstraction, record asset source/version/import method, validate asymmetric and edge grasps. (No sim versions given in the article, so its findings may predate the franka.usd variant and Isaac Sim 6.)

## 3. Existing Isaac Lab config (what it tells us)

From [`franka.py`](https://github.com/isaac-sim/IsaacLab/blob/develop/source/isaaclab_assets/isaaclab_assets/robots/franka.py) (present on `v2.3.2`, `v3.0.0-EA`, `develop`, `main`):
- `spawn.usd_path = {ISAAC_NUCLEUS_DIR}/Robots/FrankaRobotics/FrankaPanda/franka.usd`, `spawn.variants = {"Gripper": "Robotiq_2F_85"}`, `disable_gravity=True`.
- Joints: `panda_joint1..7`, driven gripper joint `finger_joint`; passive/other: `.*_inner_finger_joint`, `.*_inner_finger_knuckle_joint`, `.*_outer_.*_joint`, `right_outer_knuckle_joint` (comment: "its mimic joint").
- Actuators (implicit): arm shoulder K=1100 D=80, forearm K=1000 D=80; `gripper_drive` (`finger_joint`) K=17 D=0.02 effort 1650; `gripper_finger` (`.*_inner_finger_joint`) K=0.2 D=0.001 effort 50 ("enable the gripper to grasp in a parallel manner"); `gripper_passive` (`.*_inner_finger_knuckle_joint`, `right_outer_knuckle_joint`) K=0 D=0 ("set PD to zero for passive joints in close-loop gripper").
- Sibling: `UR10e_ROBOTIQ_2F_85_CFG` in [universal_robots.py](https://github.com/isaac-sim/IsaacLab/blob/develop/source/isaaclab_assets/isaaclab_assets/robots/universal_robots.py) with the same three-actuator pattern (finger_joint K=11.25) - a second precedent.
- Default `init_state.pos=(-0.85,0,0.76)` and the home joints (q2=-0.569, q4=-2.810, q6=3.037, q7=0.741) are demo values; override for our workcell.
- Develop branch renamed args (`joint_effort_limit` vs `effort_limit_sim`) and uses `clone()`: copy from the branch you install.
- I did not find a stock task env using `FRANKA_ROBOTIQ_GRIPPER_CFG` in the repo tree; you will write the action/obs config (e.g. a joint-position action on `finger_joint`, or `BinaryJointPositionActionCfg` with open=0 / close~0.7-0.8 rad; value UNVERIFIED, check joint limit).
- Version pairing: `develop` and `release/3.0.0` -> Isaac Sim 6.1; `main` -> Isaac Sim 4.5/5.x ([README](https://github.com/isaac-sim/IsaacLab/blob/v3.0.0-EA/README.md)). Isaac Sim releases: v6.0.0 (2026-06-04), v6.1.0 (2026-09-10), v6.2.0 (2026-10-14) ([releases](https://github.com/isaac-sim/IsaacSim/releases)); Isaac Lab develop pins 6.1, so use 6.1 until Lab moves.

## 4. Mimic joints / closed loop in Isaac Sim 6

- Articulations must be kinematic trees. [Tutorial 10: Rig Closed-Loop Structures](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/robot_setup_tutorials/rig_closed_loop_structures.html) uses the 2F-85: set `left_/right_inner_knuckle_joint` to "Exclude From Articulation" (loop break); `right_outer_knuckle_joint` gets a mimic of `finger_joint` with gearing -1 and zeroed drive; finger drives K=10 D=0.1 max force 16.5 N; inner finger joints K=0.0002 D=0.00001 to keep fingers parallel; fingertip friction 0.8 (combine = max); enable self-collision (fingers do not collide with each other by default after Onshape import); instanceable inner-finger meshes may need de-instancing to apply the physics material.
- Isaac Sim 6.0 URDF importer change ([migration guide](https://docs.isaacsim.omniverse.nvidia.com/latest/migration_guides/isaac_sim_6_0/urdf_mjcf_importer_exporter_pipeline.html)): imports now write `NewtonMimicAPI` (`newton:mimicJoint`, `newton:mimicCoef0/1`) instead of `PhysxMimicJointAPI`. Importer API is `URDFImporter(URDFImporterConfig(urdf_path, usd_path, fix_base, merge_fixed_joints, merge_mesh)).import_urdf()`; `URDFParseAndImportFile` removed; `fix_base` is tri-state (set explicitly). The page gives no closed-loop guidance and says to re-run controller tests for tendon/equality constraints. Whether the PhysX backend in Isaac Lab 3.0 honours `NewtonMimicAPI` for imported assets is UNVERIFIED: this is a reason to prefer NVIDIA's pre-authored USD (whose Multiphysics variant is "Validated") over our own URDF import.

## 5. Recommended steps

**If Panda (works today):**
1. Install Isaac Sim 6.1 + Isaac Lab `develop`/`release/3.0.0` (workstation, Docker).
2. Spawn `FRANKA_ROBOTIQ_GRIPPER_CFG` (override `init_state.pos`/`joint_pos`; check the selected variant loads, i.e. `Robotiq_2F_85` and not the default Franka Hand).
3. Print joint names and limits; confirm `finger_joint` is the driven one and mimic is active; sweep `finger_joint` 0 -> closed, check both fingers move symmetrically and no loop-violation warnings.
4. Grasp test with the 3D-printed-cube size: central, off-centre (asymmetric pad contact, ROS-I failure mode), lift; tune friction/gripper drive (the Tutorial 10 values are a starting point).
5. Fall back to the UWLab calibrated 2F-85 USD (check its license) if step 4 fails.

**If FR3:**
1. Easiest: use Panda asset + override joint limits to FR3 values (table above) - kinematics (DH) are near-identical but not guaranteed identical (UNVERIFIED: compare link lengths and flange offset in `franka_description` vs the Isaac asset). Mesh visuals are irrelevant to us since the Robot is masked out of the Scene, but collision shape/pose matter for sim.
2. Proper: compose `FrankaFR3/fr3.usd` (arm; not validated) + `Robotiq/2F-85/Robotiq_2F_85_edit.usd` (or its Multiphysics usda) via a USD variant/reference at the FR3 flange, using Isaac Sim's Robot Assembler; rely on the same actuator pattern as `FRANKA_ROBOTIQ_GRIPPER_CFG` with `fr3_joint*` names. Forum reports say Robot Assembler can be unstable with Robotiq grippers ([thread](https://forums.developer.nvidia.com/t/unstable-robot-assembler-in-isaac-sim-4-5/323628), Isaac Sim 4.5; may be fixed), so expect some manual fixed-joint authoring.
3. Alternative: generate FR3 URDF with `franka_description --no-ee`, attach the Robotiq 2F-85 URDF (e.g. [PickNik ros2_robotiq_gripper](https://github.com/PickNikRobotics/ros2_robotiq_gripper)) in one xacro, import with the 6.x importer, then break the loop and mimic by hand per Tutorial 10. Highest effort; ROS-I found this route needs "more complexity than value".
4. Real arm mounting: the real 2F-85 sits on an adapter plate on the flange; measure flange-to-TCP offset on the lab arm and match the sim TCP frame.

## 6. Open uncertainties

- Panda or FR3 in the lab (decides section 5 branch).
- Actual joint/body names and limits inside the Isaac Sim 6.1 `franka.usd` Robotiq variant (config says `finger_joint`; I did not open the USD). Does the asset need the Nucleus/S3 asset root reachable from Docker (offline pipeline requirement: pin and cache assets locally)?
- Mimic-joint behaviour under PhysX vs Newton backends in Isaac Lab 3.0; contact robustness on asymmetric cube grasps.
- Licensing: Robotiq assets are under NVIDIA's "3D Content Sharing Agreement" in the catalog; check redistribution terms before baking them into a published image.
- Physical Panda/FR3 labelling not covered by fetched Franka docs.
- Isaac Sim 6.2 (2026-10-14) vs Lab's 6.1 pin: do not upgrade Sim alone.
