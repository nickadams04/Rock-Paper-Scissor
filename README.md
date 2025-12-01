# Rock-Paper-Scissor

![Status](https://img.shields.io/badge/status-in%20development-yellow)
![ROS2](https://img.shields.io/badge/ROS2-Kilted-blue)
![License](https://img.shields.io/github/license/nickadams04/Rock-Paper-Scissor)

A single-player Rock-Paper-Scissor game against the machine, using ROS2 and Computer Vision.

## Project Status

⚠️ **Initial Development** - This is the initial commit with project architecture outlined. Implementation is in progress.

## Overview

This project implements a real-time Rock-Paper-Scissor game where a player competes against the computer using hand gestures captured through a webcam. The system uses ROS2 for inter-process communication and computer vision for hand gesture recognition.

## Architecture

The system is built on **ROS2** and consists of the following components:

### 1. Webcam Acquisition Node
- Captures real-time video feed from webcam
- Publishes image frames to ROS2 topics

### 2. Hand Gesture Recognition
- **Mediapipe.hands**: Detects and tracks hand landmarks
- **Custom CNN**: Classifies hand gestures into Rock, Paper, or Scissor

### 3. Game Logic Node
- Generates random (or guaranteed win rate) plays for the machine opponent
- Processes player gestures and determines round winners
- Maintains game state and score tracking

### 4. Visualization Node
- Annotates video frames with:
  - Detected hand landmarks
  - Recognized gesture
  - Current game state and scores
- Displays the annotated output

## Requirements

- ROS2
- Python 3
- OpenCV
- Mediapipe
- TensorFlow/PyTorch (for custom CNN)

## Installation

*Coming soon*

## Usage

*Coming soon*

## License

See [LICENSE](LICENSE) file for details.
