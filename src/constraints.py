from dataclasses import dataclass
import json


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

