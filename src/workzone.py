import math
from typing import Tuple, Union
from dataclasses import dataclass


@dataclass
class Point:
    x: Union[int, float]
    y: Union[int, float]

    def to_tuple(self) -> Tuple[int, int]:
        return self.x, self.y


def on_segment(p: Point, q: Point, r: Point) -> bool:
    """ Given three colinear points p, q, r,
        the function checks if point q lies
        on linesegment 'pr'"
    """
    if (max(p.x, r.x) >= q.x >= min(p.x, r.x)
            and max(p.y, r.y) >= q.y >= min(p.y, r.y)):
        return True
    else:
        return False


def orientation(p: Point, q: Point, r: Point) -> int:
    val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)
    if val == 0:
        return 0
    else:
        return 1 if val > 0 else 2


def do_intersec(p1: Point, q1: Point, p2: Point, q2: Point) -> bool:
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    if o1 != o2 and o3 != o4:
        return True
    elif o1 == 0 and on_segment(p1, p2, q1):
        return True
    elif o2 == 0 and on_segment(p1, q2, q1):
        return True
    elif o3 == 0 and on_segment(p2, p1, q2):
        return True
    elif o4 == 0 and on_segment(p2, q1, q2):
        return True
    else:
        return False


def is_inside(polygon: Tuple[Point, ...], p: Point) -> bool:
    if len(polygon) < 3:
        return False

    extreme = Point(math.inf, p.y)
    count = 0
    idx = 0
    while idx != -1:
        next_idx = (idx + 1) % len(polygon)
        if do_intersec(polygon[idx], polygon[next_idx], p, extreme):
            if orientation(polygon[idx], p, polygon[next_idx]) == 0:
                return on_segment(polygon[idx], p, polygon[next_idx])
            count += 1
        idx = next_idx
    return count % 2 == 1


class Workzone:
    def __init__(self, cx: int, cy: int, height: int, width: int) -> None:
        self.cx = cx
        self.cy = cy
        self.height = height
        self.width = width
        self._center = (self.cx, self.cy)

    def center(self) -> Tuple[int, int]:
        return self.cx, self.cy

    def update(self, xmin: int, ymin: int, xmax: int, ymax: int) -> None:
        self.cx = (xmax + xmin)//2
        self.cy = (ymax - ymin)//2
        self.height = ymax - ymin
        self.width = xmax - xmin

    def to_xyxy(self) -> Tuple[int, int, int, int]:
        return (self.cx - self.width//2,
                self.cy - self.height//2,
                self.cx + self.width//2,
                self.cy + self.height//2)

    def to_polygon(self) -> Tuple[Point, Point, Point, Point]:
        xyxy = self.to_xyxy()
        return ((Point(xyxy[0], xyxy[1])), (Point(xyxy[0] + self.width, xyxy[1])),
                (Point(xyxy[2], xyxy[3])), (Point(xyxy[0], xyxy[1] - self.height)))

    def contains(self, point: Tuple[int, int]) -> bool:
        return is_inside(self.to_polygon(), Point(point[0], point[1]))
