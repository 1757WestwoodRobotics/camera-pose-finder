from dataclasses import dataclass
import json


@dataclass
class CameraConstraints:
    x: float
    y: float
    z: float
    minPitch: float
    maxPitch: float
    minYaw: float
    maxYaw: float

    @staticmethod
    def fromJson(data) -> "CameraConstraints":
        return CameraConstraints(
            x=data["position"]["x"],
            y=data["position"]["y"],
            z=data["position"]["z"],
            minPitch=data["rotation"]["pitch"]["min"],
            maxPitch=data["rotation"]["pitch"]["max"],
            minYaw=data["rotation"]["yaw"]["min"],
            maxYaw=data["rotation"]["yaw"]["max"],
        )


@dataclass
class ConstraintsConfig:
    horizFOV: float
    vertFOV: float
    tagSize: float
    maxDistance: float
    minDistance: float

    cameras: list[CameraConstraints]

    @staticmethod
    def fromJson(filePath: str) -> "ConstraintsConfig":
        with open(filePath, "r") as f:
            data = json.load(f)
            return ConstraintsConfig(
                horizFOV=data["horizFOV"],
                vertFOV=data["vertFOV"],
                tagSize=data["tagSize"],
                maxDistance=data["maxDistance"],
                minDistance=data["minDistance"],
                cameras=[CameraConstraints.fromJson(cam) for cam in data["cameras"]],
            )
