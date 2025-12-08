# Acquisition Module

Status: Complete

- Captures webcam frames using OpenCV.
- Publishes frames via ROS2 on `/image_raw` using `custom_msgs/AcquisitionMsg`.
- Configurable device index and frame rate.
- Handles camera errors and graceful shutdown.

Run:
```
source ../../install/setup.bash
ros2 run acquisition acquisition_node
```
