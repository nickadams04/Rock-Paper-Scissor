# custom_msgs

Custom ROS 2 message definitions for the path planning system.

## Overview

This package defines custom message types used for communication between path planning nodes, including cone detection data, vehicle pose, waypoints, and local maps.

## Messages

### ConeStruct.msg
Represents a single detected cone in the environment.

**Fields:**
- `Point2Struct coords` - 2D position of the cone
- `uint8 color` - Color classification (0=Yellow, 1=Blue, 2=Small Orange, 3=Big Orange, 4=Unknown)

**Usage:** [TODO: Describe when and how this message is published/subscribed]

---

### Point2Struct.msg
Simple 2D point representation.

**Fields:**
- `float64 x` - X coordinate
- `float64 y` - Y coordinate

**Usage:** Used as a building block for other message types requiring 2D coordinates.

---

### PoseMsg.msg
Represents the 2D pose of the vehicle.

**Fields:**
- `Point2Struct position` - 2D position
- `float64 theta` - Heading angle (radians)

**Usage:** [TODO: Describe when and how this message is published/subscribed]

---

### LocalMapMsg.msg
Contains the local map of detected cones along with the vehicle's current pose.

**Fields:**
- `ConeStruct[] local_map` - Array of detected cones in the local frame
- `PoseMsg pose` - Current vehicle pose

**Usage:** [TODO: Describe when and how this message is published/subscribed]

---

### WaypointsMsg.msg
Path planning output containing the computed waypoints.

**Fields:**
- `uint8 count` - Number of valid waypoints in the array
- `Point2Struct[] waypoints` - Array of waypoint coordinates
- `bool is_out_of_map` - Flag indicating if the planner cannot find valid cones
- `bool should_exit` - [TODO: Describe the purpose of this flag]
- `uint32 global_index` - [TODO: Describe the purpose of this index]

**Usage:** [TODO: Describe when and how this message is published/subscribed]

---

## Dependencies

- `std_msgs` - Standard ROS 2 message types
- `geometry_msgs` - Geometry message types

## Building

This package is built as part of the workspace:

```bash
cd /path/to/workspace
colcon build --packages-select custom_msgs
source install/setup.bash
```

## License

Apache-2.0

## Maintainer

Nikos Adamopoulos (nickadamopoulos2004@gmail.com)
