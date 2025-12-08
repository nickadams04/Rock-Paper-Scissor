# Inference Module

Status: Complete

- Consumes `AcquisitionMsg`/`/image_raw`, runs Mediapipe Hands for landmarks.
- Classifies gesture into `Rock | Paper | Scissors | Unknown` with confidence.
- Publishes `GestureMsg` on `/gesture` including bbox and landmark arrays.

Run:
```
source ../../install/setup.bash
ros2 run inference inference_node
```
