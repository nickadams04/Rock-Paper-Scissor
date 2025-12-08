# Visualization Module

Status: Basic features implemented; awaiting game-sim integration

Implemented:
- Overlays hand landmarks and bounding box from `GestureMsg`.
- Displays gesture label and confidence.
- Publishes annotated frames on `/image_annotated`.
- Live preview via OpenCV window.

Pending (from `game_sim`):
- Game state and score overlay.
- Round flow indicators and timer display.

Run:
```
source ../../install/setup.bash
ros2 run visualization visualization_node
```
