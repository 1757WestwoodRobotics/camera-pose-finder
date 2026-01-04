import argparse
from functools import partial
import json
from dataclasses import dataclass
from math import exp, hypot, pi, sin
import matplotlib.pyplot as plt
from typing import List, Optional
from pathplannerlib.auto import PathPlannerAuto, RobotConfig
from pathplannerlib.path import PathPlannerTrajectoryState
from robotpy_apriltag import AprilTagFieldLayout
from wpilib import DataLogManager
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

from scipy.optimize import shgo
import multiprocessing as mp
import numpy as np
from wpiutil import wpistruct

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


def generate_score_for_camera_tag(
    camera_pose: Pose3d,
    horizFOV: float,
    vertFOV: float,
    maxDistance: float,
    apriltag_pose: Pose3d,
) -> float:
    rel = CameraTargetRelation(camera_pose, apriltag_pose)
    if not (
        abs(rel.camToTargYaw.radians()) < horizFOV / 2
        and abs(rel.camToTargPitch.radians()) < vertFOV / 2
        and abs(rel.targToCamAngle.degrees()) < 90
    ):
        # Skip tags that are not visible
        return 0

    # Compute the distance to the tag
    distance = rel.camToTargDist
    if distance > maxDistance:
        # Skip tags that are too far
        return 0

    # Compute the relative angle offset
    # Tag relative to camera's viewpoint
    horiz_angle = rel.targToCamYaw.radians()
    vert_angle = rel.targToCamPitch.radians()

    # Check if the tag is within the camera's field of view
    if abs(horiz_angle) > horizFOV / 2 or abs(vert_angle) > vertFOV / 2:
        # Skip tags that are outside the FOV
        return 0

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

    return tag_score


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


# Globals used by multiprocessing worker to access complex objects without pickling them on fork-based platforms
_MP_CONSTRAINTS: Optional[CameraConstraints] = None
_MP_TAGS: Optional[list[Pose3d]] = None
_MP_SAMPLES: Optional[list[Pose3d]] = None


def _mp_worker_score(sample, constraints: CameraConstraints, tags) -> float:
    """Worker function that scores a single sample by index using module globals.

    The worker is top-level so it can be used by multiprocessing pools. On Unix platforms
    the fork start method allows child processes to inherit the complex objects placed
    in these globals without needing to pickle them per-task.
    """
    score = generate_score_for_camera(
        poseFromArray(sample),
        constraints.horizFOV * pi / 180,
        constraints.vertFOV * pi / 180,
        constraints.maxDistance,
        [poseFromArray(tag) for tag in tags],
    )
    return score


def generate_score_for_camera(
    camera_pose: Pose3d,
    horizFOV: float,
    vertFOV: float,
    maxDistance: float,
    apriltag_poses: list[Pose3d],
) -> float:
    # Get all AprilTag poses from the field layout
    # the score for a camera is determined by a combination factors, including how many tags it can see, how close they are, and how on-axis the camera is to seeing the tag

    # filter down to only visible tags
    total_score = 0.0

    for tag in apriltag_poses:
        # Relation of the camera to the tag
        tag_score = generate_score_for_camera_tag(
            camera_pose, horizFOV, vertFOV, maxDistance, tag
        )
        # Accumulate total score
        total_score += tag_score

    return total_score


def scorePath(
    camera_transform: Transform3d,
    constraints: CameraConstraints,
    tags: list[Pose3d],
    samples: list[PathPlannerTrajectoryState],
) -> float:
    print("Testing transform", camera_transform)
    total_score = 0.0
    for sample in samples:
        camera_pose = pose2dTo3d(sample.pose) + camera_transform
        score = generate_score_for_camera(
            camera_pose,
            constraints.horizFOV * pi / 180,
            constraints.vertFOV * pi / 180,
            constraints.maxDistance,
            tags,
        )
        total_score += score
    print("score:", total_score)
    return total_score


def poseToArray(pose: Pose3d):
    return [
        pose.x,
        pose.y,
        pose.z,
        pose.rotation().x,
        pose.rotation().y,
        pose.rotation().z,
    ]


def poseFromArray(arr) -> Pose3d:
    return Pose3d(arr[0], arr[1], arr[2], Rotation3d(arr[3], arr[4], arr[5]))


def scorePath_mp(
    camera_transform: Transform3d,
    constaints: CameraConstraints,
    tags: list[Pose3d],
    samples: list[PathPlannerTrajectoryState],
    pool,
) -> float:
    print("Scoring", camera_transform)
    """Score a path by summing per-sample camera scores in parallel using multiprocessing.

    Uses a multiprocessing.Pool. On Unix, the default 'fork' start method allows child
    processes to inherit the complex objects placed into module-level globals so that
    per-task arguments remain simple (just indices) and heavy pickling is avoided.
    """

    global _MP_CONSTRAINTS, _MP_TAGS, _MP_SAMPLES
    _MP_CONSTRAINTS = constaints
    _MP_TAGS = tags
    _MP_SAMPLES = [pose2dTo3d(sample.pose) + camera_transform for sample in samples]
    # pose2d cannot be picked, process on both ends to make it so the data can be
    sample_array = [poseToArray(sample) for sample in _MP_SAMPLES]

    # Map over sample indices so only simple integers are sent to worker processes
    tag_arr = [poseToArray(tag) for tag in tags]
    score_func = partial(_mp_worker_score, constraints=constaints, tags=tag_arr)
    results = pool.map(score_func, sample_array)

    total_score = np.sum(results)

    # Clear globals to avoid holding references longer than necessary
    _MP_CONSTRAINTS = None
    _MP_TAGS = None
    _MP_SAMPLES = None
    print("Score:", total_score)
    return total_score


def objective_mp(
    x,
    constraints: CameraConstraints,
    tags: list[Pose3d],
    samples: list[PathPlannerTrajectoryState],
    pool,
):
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

    return -scorePath_mp(camera_transform, constraints, tags, samples, pool)


def objective(
    x,
    constraints: CameraConstraints,
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
    return -scorePath(
        camera_transform, constraints, tags, samples
    )  # we want to minimize, but actually maximize


def export(samples: list[PathPlannerTrajectoryState], camera_transform: Transform3d):
    # export to a log file for viewing
    DataLogManager.start()
    datalog = DataLogManager.getLog()
    # datalog writing is bugged...

    timestampId = datalog.start("/Timestamp", "int64")
    botPoseId = datalog.start("/BotPose", "struct:" + wpistruct.getTypeName(Pose2d))
    cameraPoseId = datalog.start(
        "/CameraPose", "struct:" + wpistruct.getTypeName(Pose3d)
    )

    def setupSchema(item):
        schemaId = datalog.start(
            "/.schema/struct:" + wpistruct.getTypeName(item), "structschema"
        )
        datalog.appendRaw(schemaId, wpistruct.getSchema(item).encode(), 0)

    setupSchema(Pose2d)
    setupSchema(Translation2d)
    setupSchema(Rotation2d)

    setupSchema(Pose3d)
    setupSchema(Translation3d)
    setupSchema(Rotation3d)
    setupSchema(Quaternion)

    for idx, sample in enumerate(samples):
        t = int(idx * SAMPLE_INTERVAL * 1e6)
        datalog.appendInteger(timestampId, t, t)
        datalog.appendRaw(botPoseId, wpistruct.pack(sample.pose), t)
        camera_pose = pose2dTo3d(sample.pose) + camera_transform

        datalog.appendRaw(cameraPoseId, wpistruct.pack(camera_pose), t)
        datalog.flush()

    datalog.stop()
    DataLogManager.stop()


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
    parser.add_argument(
        "--tags", dest="tags", type=str, help="Path to apriltag layout JSON file"
    )
    parser.add_argument(
        "--map",
        action="store_true",
        default=False,
        help="Display a map of all rotation values, will pick the midpoint of provided translation boundary",
    )
    parser.add_argument(
        "--mp",
        action="store_true",
        default=True,
        help="Enable multithreaded computation",
    )
    args = parser.parse_args()

    field = AprilTagFieldLayout(args.tags)
    constraints = CameraConstraints.fromJson(args.constraints)

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
        numSamples = int(totalTime / SAMPLE_INTERVAL) + 1

        for i in range(numSamples):
            t = i * SAMPLE_INTERVAL
            samples.append(idealTrajectory.sample(t))

    tagPoses = [tag.pose for tag in field.getTags()]

    # Optimize camera position and orientation

    if args.map:
        x = np.arange(constraints.minPitch, constraints.maxPitch, 5)
        y = np.arange(constraints.minYaw, constraints.maxYaw, 5)
        xgrid, ygrid = np.meshgrid(x, y)
        xy = np.stack([xgrid, ygrid])
        zgrid = np.zeros(xgrid.shape)

        for i in range(xgrid.shape[0]):
            for j in range(xgrid.shape[1]):
                cam_transform = Transform3d(
                    (constraints.minX + constraints.maxX) / 2,
                    (constraints.minY + constraints.maxY) / 2,
                    (constraints.minZ + constraints.maxZ) / 2,
                    Rotation3d(0, np.radians(i), np.radians(j)),
                )
                zgrid[i, j] = scorePath(cam_transform, constraints, tagPoses, samples)
        plt.matshow(zgrid)
        plt.show()

    else:
        bounds = [
            (constraints.minX, constraints.maxX),
            (constraints.minY, constraints.maxY),
            (constraints.minZ, constraints.maxZ),
            (constraints.minPitch, constraints.maxPitch),
            (constraints.minYaw, constraints.maxYaw),
        ]
        initial_guess = [
            (constraints.minX + constraints.maxX) / 2,
            (constraints.minY + constraints.maxY) / 2,
            (constraints.minZ + constraints.maxZ) / 2,
            (constraints.minPitch + constraints.maxPitch) / 2,
            (constraints.minYaw + constraints.maxYaw) / 2,
        ]

        # result = minimize(
        #     objective,
        #     initial_guess,
        #     args=(constraints, tagPoses, samples),
        #     bounds=bounds,
        #     method="L-BFGS-B",
        # )
        max_workers = min(32, mp.cpu_count() or 1)
        if args.mp:
            with mp.Pool(max_workers) as pool:
                result = shgo(
                    partial(objective_mp, pool=pool),
                    bounds,
                    args=(constraints, tagPoses, samples),
                    options={"disp": True, "maxiter": 100},
                )
        else:
            result = shgo(
                objective,
                bounds,
                args=(constraints, tagPoses, samples),
                options={"disp": True, "maxiter": 100},
            )
        print("Optimal Camera Position and Orientation:")
        optimal_x = result.x
        print(
            f"X: {optimal_x[0]:.2f}, Y: {optimal_x[1]:.2f}, Z: {optimal_x[2]:.2f}, Pitch: {optimal_x[3]:.2f}, Yaw: {optimal_x[4]:.2f}"
        )

        camera_transform = Transform3d(
            optimal_x[0],
            optimal_x[1],
            optimal_x[2],
            Rotation3d(
                0,
                np.radians(optimal_x[3]),
                np.radians(optimal_x[4]),
            ),
        )
        export(samples, camera_transform)


if __name__ == "__main__":
    main()
