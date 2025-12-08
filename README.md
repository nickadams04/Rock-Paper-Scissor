# Rock-Paper-Scissor

![Status](https://img.shields.io/badge/status-active-green)
![ROS2](https://img.shields.io/badge/ROS2-Kilted-blue)
![License](https://img.shields.io/github/license/nickadams04/Rock-Paper-Scissor)

A single-player Rock-Paper-Scissor game against the machine, using ROS2 and Computer Vision.

## Project Status

✅ **Progress Update (Dec 2025)**
- Acquisition: complete and stable (ROS2 image streaming)
- Inference: complete (gesture detection + classification)
- Visualization: basic overlay working; awaiting game-sim integration
- Game Simulation: in progress (scores/round flow to be wired)

## Overview

This project implements a real-time Rock-Paper-Scissor game where a player competes against the computer using hand gestures captured through a webcam. The system uses ROS2 for inter-process communication and computer vision for hand gesture recognition.

## Architecture

The system is built on **ROS2** and consists of the following components:

### 1. Webcam Acquisition Node (complete)
- Captures real-time video feed from webcam
- Publishes frames on `/image_raw` (custom `AcquisitionMsg`)
- Handles device selection and frame rate control

### 2. Hand Gesture Recognition (complete)
- Mediapipe Hands: landmark detection and bounding box
- Classifier: maps landmarks to `Rock | Paper | Scissors | Unknown`
- Publishes on `/gesture` (`GestureMsg`) with confidence scores

### 3. Game Logic Node
- Generates random (or guaranteed win rate) plays for the machine opponent
- Processes player gestures and determines round winners
- Maintains game state and score tracking

### 4. Visualization Node (ongoing)
- Overlays landmarks, bounding box, and gesture + confidence
- Publishes annotated frames on `/image_annotated`
- Displays a window (`cv2.imshow`) for live preview
- Pending: game state + scores overlay from `game_sim`

## Requirements

- ROS2 (Humble or later recommended)
- Python 3.10+
- See `ROS_Workspace/requirements.txt` for ROS package dependencies

## Installation

1) Create and source ROS2 workspace
```
cd ROS_Workspace
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
2) Build ROS packages
```
cd ROS_Workspace
colcon build
source install/setup.bash
```

## Usage

- Launch nodes (example; adjust to your environment):
```
source ROS_Workspace/install/setup.bash
ros2 run acquisition acquisition_node
ros2 run inference inference_node
ros2 run visualization visualization_node
ros2 launch acquisition all.launch.py
```
- Optional scripts are in `scripts/` (e.g., `launchAll.sh`).

## License

See [LICENSE](LICENSE) file for details.
