# Splat2Sim — Context

## Glossary

- **Capture**: A raw handheld walk-around recording of the lab from a RealSense D455 (color, depth, IMU), stored as a bag/MCAP file.
- **Scene**: The reconstructed lab as USD — a photoreal visual layer plus a simplified collision layer — loadable in Isaac Sim / Isaac Lab. Static background only: nothing in it moves.
- **Task environment**: A Scene plus robot(s) and manipulated objects, configured for training or evaluating a policy.
- **Manipulated object**: A movable rigid object added to a Scene from a separate source; never carved out of the Scene's reconstruction (in v1).
- **Robot**: In v1, the lab's Franka Research 3 (FR3) arm fitted with a Robotiq 2F-85 gripper (not the stock Franka Hand). Bolted down, so it appears in every Capture.
- **Robot mask**: The pixels of the real Robot in a Capture, removed before reconstruction so the Scene doesn't contain a ghost robot.
- **Anchor**: The physical marker (calibration board) whose known pose relative to the Robot base fixes the Scene's metric scale and its alignment to the robot.
- **Policy cameras**: The two real cameras the policies see through — wrist-mounted and a fixed front camera with an angled top-down view. The Scene must look right from their viewpoints.
