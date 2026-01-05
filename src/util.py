from wpimath.geometry import Pose2d, Pose3d, Rotation3d


def pose2dTo3d(pose: Pose2d) -> Pose3d:
    return Pose3d(pose.X(), pose.Y(), 0, Rotation3d(0, 0, pose.rotation().radians()))


def poseToArray(pose: Pose3d):
    return [
        pose.x,
        pose.y,
        pose.z,
        pose.rotation().x,
        pose.rotation().y,
        pose.rotation().z,
    ]


def arrayToPose(arr) -> Pose3d:
    return Pose3d(arr[0], arr[1], arr[2], Rotation3d(arr[3], arr[4], arr[5]))
