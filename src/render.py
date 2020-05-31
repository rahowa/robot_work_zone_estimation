import cv2
import numpy as np
from nptyping import Array
from typing import Sequence, Tuple, Union

from src.obj_loader import OBJ
from src.utills import hex_to_rgb


class RenderZone:
    def __init__(self, obj: OBJ, scale_factor: float, marker_shape: Tuple[int, int]) -> None:
        self.obj = obj
        self.scale_factor = scale_factor
        self.marker_shape = marker_shape

    def render(self,
            img: Array[int],
            projection: Array[float],
            color: Union[bool, Sequence[int]] = False) -> Array[int]:
        vertices = self.obj.vertices
        scale_matrix = np.eye(3) * self.scale_factor
        h, w = self.marker_shape
        tmp_image = np.zeros_like(img)

        for face in self.obj.faces:
            face_vertices = face[0]
            points = np.array([vertices[vertex - 1] for vertex in face_vertices])
            points = np.dot(points, scale_matrix)
            # render model in the middle of the reference surface. To do so,
            # model points must be displaced
            points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
            dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
            imgpts = np.int32(dst)
            if color is False:
                cv2.fillConvexPoly(tmp_image, imgpts, (137, 27, 211))
            else:
                color = hex_to_rgb(face[-1])
                color = color[::-1] # reverse
                cv2.fillConvexPoly(tmp_image, imgpts, color)
        return np.uint8(np.where(tmp_image == 0, 1, 0.5) * img) + tmp_image//2