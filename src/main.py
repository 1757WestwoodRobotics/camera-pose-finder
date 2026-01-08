import argparse
from dataclasses import dataclass
from functools import partial, reduce
from operator import add
from math import cos, exp, hypot, pi, sin
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from typing import List, Optional
import numpy.typing as npt
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
from constraints import CameraConstraints, ConstraintsConfig
from util import mapRange, pose2dTo3d, poseToArray, arrayToPose

SAMPLE_INTERVAL = 0.01  # seconds


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


@dataclass
class CameraTagScore:
    orientation_score: float = 0
    fov_score: float = 0
    distance: float = 0
    can_see: bool = False


def generate_score_for_camera_tag(
    camera_pose: Pose3d,
    horizFOV: float,
    vertFOV: float,
    maxDistance: float,
    minDistance: float,
    tagSize: float,
    apriltag_pose: Pose3d,
) -> CameraTagScore:
    rel = CameraTargetRelation(camera_pose, apriltag_pose)
    if not (
        abs(rel.camToTargYaw.radians()) < horizFOV / 2
        and abs(rel.camToTargPitch.radians()) < vertFOV / 2
        and abs(rel.targToCamAngle.degrees()) < 90
    ):
        # Skip tags that are not visible
        return CameraTagScore(can_see=False)

    # Compute the distance to the tag
    distance = max(
        rel.camToTargDist, minDistance
    )  # no better information comes from being closer than min distance
    if distance > maxDistance:
        # Skip tags that are too far
        return CameraTagScore(can_see=False)

    # Compute the relative angle offset
    # Tag relative to camera's viewpoint
    horiz_angle = rel.targToCamYaw.radians()
    vert_angle = rel.targToCamPitch.radians()
    # Check if the tag is within the camera's field of view
    if abs(horiz_angle) > horizFOV / 2 or abs(vert_angle) > vertFOV / 2:
        # Skip tags that are outside the FOV
        return CameraTagScore(can_see=False)

    halfTag = tagSize / 2
    upperLeft = (
        rel.camToTarg + Transform3d(halfTag, 0, -halfTag, Rotation3d())
    ).translation()
    upperRight = (
        rel.camToTarg + Transform3d(halfTag, 0, halfTag, Rotation3d())
    ).translation()
    lowerLeft = (
        rel.camToTarg + Transform3d(-halfTag, 0, -halfTag, Rotation3d())
    ).translation()
    lowerRight = (
        rel.camToTarg + Transform3d(-halfTag, 0, halfTag, Rotation3d())
    ).translation()
    # check if each point is within camera's fov
    for point in [upperLeft, upperRight, lowerLeft, lowerRight]:
        horiz_angle = Rotation2d(point.x, point.y).radians()
        dist_xy = hypot(point.x, point.y)
        vert_angle = Rotation2d(dist_xy, -point.z).radians()

        if abs(horiz_angle) > horizFOV / 2 or abs(vert_angle) > vertFOV / 2:
            # Skip tags that are outside the FOV
            return CameraTagScore(can_see=False)

    # Compute the orientation alignment score for the tag
    # Tags directly facing the camera are "on-axis"
    # orientation_score = cos(rel.targToCamAngle.radians()) ** 2
    orientation_score = pow(
        (1 - pow(2 * rel.targToCamAngle.radians() / pi, 2)), 0.4
    )  # tweak the 0.4 for how tolerant being off-axis is

    # Gaussian falloff for each axis based on distance from FOV center
    horiz_falloff = exp(-(horiz_angle**4) / (2 * (horizFOV / 4) ** 2))
    vert_falloff = exp(-(vert_angle**4) / (2 * (vertFOV / 4) ** 2))
    # horiz_falloff = pow(1 - (2 * horiz_angle / horizFOV) ** 2, 0.5)
    # vert_falloff = pow(1 - (2 * vert_angle / vertFOV) ** 2, 0.5)
    fov_score = horiz_falloff * vert_falloff

    return CameraTagScore(orientation_score, fov_score, distance, True)


# Globals used by multiprocessing worker to access complex objects without pickling them on fork-based platforms
_MP_CONSTRAINTS: Optional[ConstraintsConfig] = None
_MP_TAGS: Optional[list[Pose3d]] = None


def _mp_worker_score(sample, constraints: ConstraintsConfig) -> float:
    """Worker function that scores a single sample by index using module globals.

    The worker is top-level so it can be used by multiprocessing pools. On Unix platforms
    the fork start method allows child processes to inherit the complex objects placed
    in these globals without needing to pickle them per-task.
    """
    score = generate_score_for_camera(
        arrayToPose(sample),
        constraints.horizFOV * pi / 180,
        constraints.vertFOV * pi / 180,
        constraints.maxDistance,
        constraints.minDistance,
        constraints.tagSize,
        _MP_TAGS,
    )
    return score


def generate_score_for_camera(
    camera_pose: Pose3d,
    horizFOV: float,
    vertFOV: float,
    maxDistance: float,
    minDistance: float,
    tagSize: float,
    apriltag_poses: list[Pose3d],
) -> float:
    # Get all AprilTag poses from the field layout
    # the score for a camera is determined by a combination factors, including how many tags it can see, how close they are, and how on-axis the camera is to seeing the tag

    # filter down to only visible tags
    total_score = 0.0
    total_distance = 0.0
    tag_total = 0

    for tag in apriltag_poses:
        # Relation of the camera to the tag
        tag_score = generate_score_for_camera_tag(
            camera_pose, horizFOV, vertFOV, maxDistance, minDistance, tagSize, tag
        )
        if tag_score.can_see:
            tag_total += 1
            total_distance += tag_score.distance
            total_score += tag_score.fov_score * tag_score.orientation_score
            # total_score += tag_score.fov_score
            # total_score += 1
        else:
            continue
    # use distances and total tags to prefer closer of more tags
    if tag_total == 0:
        return 0

    average_distance = total_distance / tag_total
    total_score *= tag_total / (pow(average_distance, 2))

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
            constraints.minDistance,
            constraints.tagSize,
            tags,
        )
        total_score += score
    print("score:", total_score)
    return total_score


def scorePath_mp(
    camera_transform: Transform3d,
    samples: list[PathPlannerTrajectoryState],
    pool,
) -> float:
    """Score a path by summing per-sample camera scores in parallel using multiprocessing.

    Uses a multiprocessing.Pool. On Unix, the default 'fork' start method allows child
    processes to inherit the complex objects placed into module-level globals so that
    per-task arguments remain simple (just indices) and heavy pickling is avoided.
    """

    _MP_SAMPLES = [pose2dTo3d(sample.pose) + camera_transform for sample in samples]
    # pose2d cannot be picked, process on both ends to make it so the data can be
    sample_array = [poseToArray(sample) for sample in _MP_SAMPLES]

    # Map over sample indices so only simple integers are sent to worker processes
    score_func = partial(_mp_worker_score, constraints=_MP_CONSTRAINTS)
    results = pool.map(score_func, sample_array)

    total_score = np.sum(results)
    return total_score


def objective_mp(
    x,
    samples: list[PathPlannerTrajectoryState],
    position: Translation3d,
    pool,
):
    camera_transform = Transform3d(
        position,
        Rotation3d(
            0,
            np.radians(x[0]),
            np.radians(x[1]),
        ),
    )

    return -scorePath_mp(camera_transform, samples, pool)


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


def objective_multi_mp(
    x,
    positions: list[Translation3d],
    samples: list[PathPlannerTrajectoryState],
    pool,
) -> float:
    camera_transforms = [
        Transform3d(
            pos, Rotation3d(0, np.radians(x[idx * 2]), np.radians(x[idx * 2 + 1]))
        )
        for idx, pos in enumerate(positions)
    ]
    totalScore = 0
    for transform in camera_transforms:
        totalScore -= scorePath_mp(transform, samples, pool)

    return totalScore


def export(
    samples: list[PathPlannerTrajectoryState], camera_transforms: list[Transform3d]
):
    # export to a log file for viewing
    DataLogManager.start()
    datalog = DataLogManager.getLog()
    # datalog writing is bugged...

    timestampId = datalog.start("/Timestamp", "int64")
    botPoseId = datalog.start("/BotPose", "struct:" + wpistruct.getTypeName(Pose2d))
    cameraPoseIds = []
    for idx, _ in enumerate(camera_transforms):
        cameraPoseIds.append(
            datalog.start(
                f"/CameraPose{idx}", "struct:" + wpistruct.getTypeName(Pose3d)
            )
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
        for idx, camera_transform in enumerate(camera_transforms):
            camera_pose = pose2dTo3d(sample.pose) + camera_transform

            datalog.appendRaw(cameraPoseIds[idx], wpistruct.pack(camera_pose), t)
        datalog.flush()

    datalog.stop()
    DataLogManager.stop()


def solve_cameras(
    tagPoses: list[Pose3d],
    constraints: ConstraintsConfig,
    samples: list[PathPlannerTrajectoryState],
) -> tuple[list[Transform3d], npt.NDArray]:
    bounds = [
        [
            (constraint.minPitch, constraint.maxPitch),
            (constraint.minYaw, constraint.maxYaw),
        ]
        for constraint in constraints.cameras
    ]
    bounds = reduce(add, bounds, [])
    print("Bounds:", bounds)

    positions = [Translation3d(co.x, co.y, co.z) for co in constraints.cameras]
    print("Positions:", positions)

    global _MP_TAGS, _MP_CONSTRAINTS
    _MP_TAGS = tagPoses
    _MP_CONSTRAINTS = constraints
    max_workers = min(32, mp.cpu_count() or 1)
    with mp.Pool(max_workers) as pool:
        result = shgo(
            partial(
                objective_multi_mp, pool=pool, samples=samples, positions=positions
            ),
            bounds,
            n=100,
            iters=5,
            sampling_method="sobol",
        )
    # Clear globals to avoid holding references longer than necessary
    _MP_CONSTRAINTS = None
    _MP_TAGS = None
    optimal_x = result.x
    print(optimal_x)

    camera_transforms = [
        Transform3d(pos, Rotation3d(0, np.radians(pitch), np.radians(yaw)))
        for pos, pitch, yaw in zip(positions, optimal_x[::2], optimal_x[1:][::2])
    ]
    return camera_transforms, result.xl


def solve_camera(
    tagPoses: list[Pose3d],
    constraints: ConstraintsConfig,
    camidx: int,
    samples: list[PathPlannerTrajectoryState],
) -> tuple[Transform3d, npt.NDArray]:
    cam = constraints.cameras[camidx]
    bounds = [
        (cam.minPitch, cam.maxPitch),
        (cam.minYaw, cam.maxYaw),
    ]
    print("Bounds:", bounds)

    position = Translation3d(cam.x, cam.y, cam.z)
    print("Position:", position)

    global _MP_TAGS, _MP_CONSTRAINTS
    _MP_TAGS = tagPoses
    _MP_CONSTRAINTS = constraints
    max_workers = min(32, mp.cpu_count() or 1)
    with mp.Pool(max_workers) as pool:
        result = shgo(
            partial(objective_mp, pool=pool, samples=samples, position=position),
            bounds,
            # n=100,
            # iters=5,
            # sampling_method="sobol",
        )
    # Clear globals to avoid holding references longer than necessary
    _MP_CONSTRAINTS = None
    _MP_TAGS = None
    optimal_x = result.x
    print(optimal_x)

    camera_transform = Transform3d(
        position, Rotation3d(0, np.radians(optimal_x[0]), np.radians(optimal_x[1]))
    )
    return camera_transform, result.xl


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
        "--map-resolution",
        default=5,
        dest="mres",
        type=int,
        help="The resolution to use, in degrees, when generating a map",
    )
    parser.add_argument(
        "--plotoptimal",
        action="store_true",
        default=False,
        help="Plot the optimal locations, also includes map",
    )
    parser.add_argument(
        "--seperate",
        action="store_true",
        default=False,
        help="Seperate each camera and solve individually, as opposed to a combined camera solution",
    )
    args = parser.parse_args()

    field = AprilTagFieldLayout(args.tags)
    constraints = ConstraintsConfig.fromJson(args.constraints)

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

    locallimits = np.zeros((1, 1))
    if (not args.map) or args.plotoptimal:
        if args.seperate:
            optimal_transforms = []
            locallimits = []
            for i, _ in enumerate(constraints.cameras):
                optimal_transform, locallimit = solve_camera(
                    tagPoses, constraints, i, samples
                )
                optimal_transforms.append(optimal_transform)
                locallimits.extend(locallimit)
            locallimits = np.array(locallimits)
            print(locallimits)
        else:
            optimal_transforms, locallimits = solve_cameras(
                tagPoses, constraints, samples
            )
            print(locallimits)
        print("Optimal Camera Position and Orientation:")
        for optimal_transform in optimal_transforms:
            print(
                f"X: {optimal_transform.x:.2f}, Y: {optimal_transform.y:.2f}, Z: {optimal_transform.z:.2f}, Pitch: {optimal_transform.rotation().y_degrees:.2f}, Yaw: {optimal_transform.rotation().z_degrees:.2f}"
            )
        export(samples, optimal_transforms)

    if args.map:
        x = np.arange(
            constraints.cameras[0].minPitch, constraints.cameras[0].maxPitch, args.mres
        )
        y = np.arange(
            constraints.cameras[0].minYaw, constraints.cameras[0].maxYaw, args.mres
        )
        xgrid, ygrid = np.meshgrid(x, y)
        zgrid = np.zeros((len(constraints.cameras), *xgrid.shape))

        global _MP_TAGS, _MP_CONSTRAINTS
        _MP_TAGS = tagPoses
        _MP_CONSTRAINTS = constraints
        max_workers = min(32, mp.cpu_count() or 1)
        with mp.Pool(max_workers) as pool:
            for camidx, camera in enumerate(constraints.cameras):
                print("Computing camera", camidx)
                for i in range(xgrid.shape[0]):
                    for j in range(xgrid.shape[1]):
                        cam_transform = Transform3d(
                            camera.x,
                            camera.y,
                            camera.z,
                            Rotation3d(
                                0, np.radians(xgrid[i, j]), np.radians(ygrid[i, j])
                            ),
                        )
                        zgrid[camidx, i, j] = scorePath_mp(cam_transform, samples, pool)

        fig = plt.figure()
        for i in range(zgrid.shape[0]):
            ax = fig.add_subplot(zgrid.shape[0], 1, i + 1)
            im = ax.imshow(zgrid[i].T, interpolation="bilinear", cmap="turbo")
            ax.set_xlabel("yaw")
            ax.set_ylabel("pitch")
            ax.set_title(f"Camera {i}")

            ticks_x = ticker.FuncFormatter(
                lambda x, pos: "{0:g}".format(
                    mapRange(
                        x,
                        0,
                        zgrid.shape[1],
                        constraints.cameras[i].minYaw,
                        constraints.cameras[i].maxYaw,
                    )
                )
            )
            ax.xaxis.set_major_formatter(ticks_x)

            ticks_y = ticker.FuncFormatter(
                lambda y, pos: "{0:g}".format(
                    mapRange(
                        y,
                        0,
                        zgrid.shape[2],
                        constraints.cameras[i].minPitch,
                        constraints.cameras[i].maxPitch,
                    )
                )
            )
            ax.yaxis.set_major_formatter(ticks_y)

            if args.plotoptimal:
                for j in range(locallimits.shape[0]):
                    ax.plot(
                        mapRange(
                            locallimits[j, 2 * i + 1],
                            constraints.cameras[i].minYaw,
                            constraints.cameras[i].maxYaw,
                            0,
                            zgrid.shape[1],
                        ),
                        mapRange(
                            locallimits[j, 2 * i],
                            constraints.cameras[i].minPitch,
                            constraints.cameras[i].maxPitch,
                            0,
                            zgrid.shape[2],
                        ),
                        "ro",
                        ms=2,
                    )

        plt.show()


if __name__ == "__main__":
    main()
