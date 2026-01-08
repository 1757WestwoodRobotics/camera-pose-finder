# Camera Location Optimizer

This project provides a tool used to optimize the position and angle of a camera on an FRC robot to maximize accurate vision target detection. Users provide the constaints for camera location, a path to follow from pathplanner, and a tag set for vision targets. The tool then simulates various camera positions and angles to determine the optimal setup for the robot's vision system.

## Features
- Define camera location constraints (x, y, z, pitch, yaw, distance)
- Input pathplanner paths for robot movement simulation
- Specify vision target tag sets via AprilTagFieldLayout JSON files

## Usage
1. Clone the repository:
```bash
git clone https://github.com/1757WestwoodRobotics/camera-pose-finder
```
2. Install dependencies:
```bash
uv sync 
```
3. Construct a constraints file in JSON format specifying camera locations and angles to test. 

An example file exists at `constraints_multi.json`.

4. Construct a apriltag field layout JSON file specifying the vision targets. An example file exists at `2025-reefscape-no-barge.json`.

5. Open this project in PathPlanner and construct an auto routine that uses one or more paths.

6. Run the optimizer with the following command:
```bash
uv run src/main.py --constraints path/to/constraints.json --path name_of_auto_routine --tags path/to/apriltag/fieldlayout.json
```

For example
```bash
uv run src/main.py --constraints constraints_multi.json --path "dual approach" --tags 2025-reefscape-no-barge.json
```

A couple of additional arguments exists:
- `--map` generates an image visualization of the "score" across all possible camera angles
    - `--map-resolution` can be used to specify with what resolution to render the map. The default is to take a sample every 5 degrees
- `--plotoptimal` when combined with `--map` will also compute the complete optimal camera location and plot it on the generated map
- `--seperate` can be used to generate seperate results for each camera's angles, compared to the overall optimal. This may be faster if there are many cameras to test

## Output
By default, upon completing any ideal computation, a wpilog file will be created. This file can be opened in AdvantageScope to visualize the results of the optimization. It is reccomended to view each camera pose as a camera override.
