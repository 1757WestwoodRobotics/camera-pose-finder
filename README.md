# Camera Location Optimizer

This project provides a tool used to optimize the position and angle of a camera on an FRC robot to maximize accurate vision target detection. Users provide the constaints for camera location, a path to follow from pathplanner, and a tag set for vision targets. The tool then simulates various camera positions and angles to determine the optimal setup for the robot's vision system.

## Features
- Define camera location constraints (x, y, z, pitch, yaw, distance)
- Input pathplanner paths for robot movement simulation
- Specify vision target tag sets via AprilTagFieldLayout JSON files

## Usage
1. Clone the repository:
```bash
git clone
```
2. Install dependencies:
```bash
uv sync 
```
3. Construct a constraints file in JSON format specifying camera location limits. 

An example file exists at `constraints.json`.

4. Construct a apriltag field layout JSON file specifying the vision targets. An example file exists at `2025-reefscape-no-barge.json`.

5. Open this project in PathPlanner and construct an auto routine that uses one or more paths.

6. Run the optimizer with the following command:
```bash
uv run main.py --constraints path/to/constraints.json --path name_of_auto_routine --tags path/to/apriltag/fieldlayout.json
```

For example
```bash
uv run main.py --constraints constraints.json --path "3 piece right" --tags 2025-reefscape-no-barge.json
```
