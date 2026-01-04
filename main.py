import argparse
import json
import matplotlib.pyplot as plt
from dataclasses import dataclass
from math import atan2, exp, hypot, pi, sin
from typing import List
from pathplannerlib.auto import PathPlannerAuto, RobotConfig
from pathplannerlib.path import PathPlannerTrajectoryState
from robotpy_apriltag import AprilTagFieldLayout, AprilTagField
from scipy.sparse import data
from wpimath.geometry import (
    Pose2d,
    Pose3d,
    Quaternion,
    Rotation2d,
    Rotation3d,
    Transform3d,
    Translation2d,
    Translation3d,
)

from scipy.optimize import minimize, differential_evolution
import numpy as np
import multiprocessing as mp
from wpiutil import DataLogWriter, wpistruct

SAMPLE_INTERVAL = 0.02  # seconds


def pose2dTo3d(pose: Pose2d) -> Pose3d:
    return Pose3d(pose.X(), pose.Y(), 0, Rotation3d(0, 0, pose.rotation().radians()))


class CameraTargetRelation:
    def __init__(self, cameraPose: Pose3d, targetPose: Pose3d) -> None:
        self.cameraPose = cameraPose
        self.camToTarg = Transform3d(cameraPose, targetPose)
        self.camToTargDist = self.camToTarg.translation().norm()
        self.camToTargDistXY = hypot(
            self.camToTarg.translation().X(), self.camToTarg.translation().Y()
        )
        self.camToTargYaw = Rotation2d(self.camToTarg.X(), self.camToTarg.Y())
        self.camToTargPitch = Rotation2d(self.camToTargDistXY, -self.camToTarg.Z())
        self.camToTargAngle = Rotation2d(
            hypot(self.camToTargYaw.radians(), self.camToTargPitch.radians())
        )

        self.targToCam = Transform3d(targetPose, cameraPose)
        self.targToCamYaw = Rotation2d(self.targToCam.X(), self.targToCam.Y())
        self.targToCamPitch = Rotation2d(self.camToTargDistXY, -self.targToCam.Z())
        self.targToCamAngle = Rotation2d(
            hypot(self.targToCamYaw.radians(), self.targToCamPitch.radians())
        )


def generate_score_for_camera(
    robot_pose: Pose2d,
    camera_transform: Transform3d,
    horizFOV: float,
    vertFOV: float,
    maxDistance: float,
    apriltag_poses: list[Pose3d],
) -> float:
    # Get all AprilTag poses from the field layout
    # the score for a camera is determined by a combination factors, including how many tags it can see, how close they are, and how on-axis the camera is to seeing the tag

    # filter down to only visible tags
    camera_pose = pose2dTo3d(robot_pose) + (camera_transform)

    total_score = 0.0

    for tag in apriltag_poses:
        # Relation of the camera to the tag
        rel = CameraTargetRelation(camera_pose, tag)

        # Compute the distance to the tag
        distance = rel.camToTargDist
        # if distance > maxDistance:
        #     # Skip tags that are too far
        #     continue

        # Compute the relative angle offset
        # Tag relative to camera's viewpoint
        horiz_angle = rel.targToCamYaw.radians()
        vert_angle = rel.targToCamPitch.radians()

        # Check if the tag is within the camera's field of view
        if abs(horiz_angle) > horizFOV / 2 or abs(vert_angle) > vertFOV / 2:
            # Skip tags that are outside the FOV
            continue

        # Compute the orientation alignment score for the tag
        # Tags directly facing the camera are "on-axis"
        orientation_score = sin(rel.targToCamAngle.radians()) ** 2

        # Gaussian falloff for each axis based on distance from FOV center
        horiz_falloff = exp(-(horiz_angle**2) / (2 * (horizFOV / 4) ** 2))
        vert_falloff = exp(-(vert_angle**2) / (2 * (vertFOV / 4) ** 2))
        fov_score = horiz_falloff * vert_falloff

        # Distance score is inversely proportional to distance squared
        distance_score = 1 / (distance**2)

        # Combine factors
        tag_score = distance_score * fov_score * (1 - orientation_score)

        # Accumulate total score
        total_score += tag_score

    return total_score


@dataclass
class CameraConstraints:
    horizFOV: float
    vertFOV: float
    maxDistance: float
    minX: float
    maxX: float
    minY: float
    maxY: float
    minZ: float
    maxZ: float
    minPitch: float
    maxPitch: float
    minYaw: float
    maxYaw: float

    @staticmethod
    def fromJson(filePath: str) -> "CameraConstraints":

        with open(filePath, "r") as f:
            data = json.load(f)
            return CameraConstraints(
                horizFOV=data["horizFOV"],
                vertFOV=data["vertFOV"],
                maxDistance=data["maxDistance"],
                minX=data["minX"],
                maxX=data["maxX"],
                minY=data["minY"],
                maxY=data["maxY"],
                minZ=data["minZ"],
                maxZ=data["maxZ"],
                minPitch=data["minPitch"],
                maxPitch=data["maxPitch"],
                minYaw=data["minYaw"],
                maxYaw=data["maxYaw"],
            )


def scorePath(
    camera_transform: Transform3d,
    constaints: CameraConstraints,
    tags: list[Pose3d],
    samples: list[PathPlannerTrajectoryState],
) -> float:
    total_score = 0.0
    for sample in samples:
        score = generate_score_for_camera(
            sample.pose,
            camera_transform,
            constaints.horizFOV * pi / 180,
            constaints.vertFOV * pi / 180,
            constaints.maxDistance,
            tags,
        )
        if score != float("inf"):
            total_score += score
    if total_score == 0.0:
        return float("inf")
    return total_score


def objective(
    x,
    constaints: CameraConstraints,
    tags: list[Pose3d],
    samples: list[PathPlannerTrajectoryState],
) -> float:
    camera_transform = Transform3d(
        x[0],
        x[1],
        x[2],
        Rotation3d(
            0,
            np.radians(x[3]),
            np.radians(x[4]),
        ),
    )
    return scorePath(camera_transform, constaints, tags, samples)


def main():
    parser = argparse.ArgumentParser(description="Camera Location Optimizer")
    parser.add_argument(
        "--version",
        action="version",
        version="cameraoptimizer version 1.0.0",
    )
    parser.add_argument(
        "--path",
        dest="path",
        type=str,
        help="Path to the input path file to optimize camera",
    )
    parser.add_argument(
        "--constraints",
        dest="constraints",
        type=str,
        help="Path to the constraints file",
    )
    args = parser.parse_args()

    field = AprilTagFieldLayout.loadField(AprilTagField.k2025ReefscapeAndyMark)
    constaints = CameraConstraints.fromJson(args.constraints)

    # get the robot config from the file settings
    config = RobotConfig.fromGUISettings()
    paths = PathPlannerAuto.getPathGroupFromAutoFile(args.path)

    samples: List[PathPlannerTrajectoryState] = []
    for path in paths:
        idealTrajectory = path.getIdealTrajectory(config)
        if idealTrajectory is None:
            print("Could not generate ideal trajectory for path:", path.name)
            continue
        totalTime = idealTrajectory.getTotalTimeSeconds()
        print(f"Path: {path.name}, Total Time: {totalTime:.2f} seconds")
        numSamples = int(totalTime / SAMPLE_INTERVAL) + 1

        for i in range(numSamples):
            t = i * SAMPLE_INTERVAL
            samples.append(idealTrajectory.sample(t))

    # Sample the trajectory and compute scores

    # Optimize camera position and orientation

    bounds = [
        (constaints.minX, constaints.maxX),
        (constaints.minY, constaints.maxY),
        (constaints.minZ, constaints.maxZ),
        (constaints.minPitch, constaints.maxPitch),
        (constaints.minYaw, constaints.maxYaw),
    ]
    initial_guess = [
        (constaints.minX + constaints.maxX) / 2,
        (constaints.minY + constaints.maxY) / 2,
        (constaints.minZ + constaints.maxZ) / 2,
        (constaints.minPitch + constaints.maxPitch) / 2,
        (constaints.minYaw + constaints.maxYaw) / 2,
    ]

    tagPoses = [tag.pose for tag in field.getTags()]

    fig = plt.figure()
    x = np.arange(constaints.minPitch, constaints.maxPitch)
    y = np.arange(constaints.minYaw, constaints.maxYaw)
    xgrid, ygrid = np.meshgrid(x, y)
    xy = np.stack([xgrid, ygrid])
    # go through all points and create a 2d numpy array
    zgrid = np.zeros(xgrid.shape)
    print(xgrid.shape, xy.shape)
    totalSamples = xgrid.shape[0] * xgrid.shape[1]
    c = 0
    for i in range(xgrid.shape[0]):
        for j in range(xgrid.shape[1]):
            c+=1
            zgrid[i, j] = objective(
                [
                    initial_guess[0],
                    initial_guess[1],
                    initial_guess[2],
                    xy[0, i, j],
                    xy[1, i, j],
                ],
                constaints,
                tagPoses,
                samples,
            )
            print(zgrid[i,j])
            print(c / totalSamples)

    ax = fig.add_subplot(111)
    im = ax.imshow(
        zgrid,
        extent=(
            constaints.minPitch,
            constaints.maxPitch,
            constaints.minYaw,
            constaints.maxYaw,
        ),
        origin="lower",
        aspect="auto",
    )
    plt.show()

    # camera_transform = Transform3d(
    #     optimal_x[0],
    #     optimal_x[1],
    #     optimal_x[2],
    #     Rotation3d(
    #         0,
    #         np.radians(optimal_x[3]),
    #         np.radians(optimal_x[4]),
    #     ),
    # )

    # export to a log file for viewing

    # datalog = DataLogWriter("OUTPUT.wpilog")

    # timestampId = datalog.start("/Timestamp", "int64")
    # botPoseId = datalog.start("/BotPose", "struct:" + wpistruct.getTypeName(Pose2d))
    # cameraPoseId = datalog.start(
    #     "/CameraPose", "struct:" + wpistruct.getTypeName(Pose3d)
    # )

    # def setupSchema(item):
    #     schemaId = datalog.start(
    #         "/.schema/struct:" + wpistruct.getTypeName(item), "structschema"
    #     )
    #     datalog.appendRaw(schemaId, wpistruct.getSchema(item).encode(), 0)

    # setupSchema(Pose2d)
    # setupSchema(Translation2d)
    # setupSchema(Rotation2d)

    # setupSchema(Pose3d)
    # setupSchema(Translation3d)
    # setupSchema(Rotation3d)
    # setupSchema(Quaternion)

    # for idx, sample in enumerate(samples):
    #     t = int(idx * SAMPLE_INTERVAL * 1e6)
    #     datalog.appendInteger(timestampId, t, t)
    #     datalog.appendRaw(botPoseId, wpistruct.pack(sample.pose), t)
    #     camera_pose = pose2dTo3d(sample.pose) + camera_transform

    #     datalog.appendRaw(cameraPoseId, wpistruct.pack(camera_pose), t)

    #     datalog.flush()

    # datalog.stop()


if __name__ == "__main__":
    main()
