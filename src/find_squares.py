import cv2
import numpy as np

# def find_cosine(point_low, point_middle, point_high):
#     vec1 = np.array([point_low, point_middle])
#     vec2 = np.array([point_middle, point_high])
#     return np.dot(vec1, vec2)/np.linalg.norm(vec1)/np.linalg.norm(vec2)


def find_cosine(point_b, point_c, point_a):
    x, y = 0, 1
    dx1 = point_b[:, x] - point_a[:, x]
    dy1 = point_b[:, y] - point_a[:, y]
    dx2 = point_c[:, x] - point_a[:, x]
    dy2 = point_c[:, y] - point_a[:, y]
    return (dx1*dx2 + dy1*dy2) / np.sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-8)


class SquareFinder:
    """
    Original implementation: https://github.com/tentone/aruco
    """
    def __init__(self, cosine_limit: float, min_area: float):
        self.cosine_limit = cosine_limit
        self.min_area = min_area

    def __call__(self, image):
        squares = list()
        contours = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
        for cnt in contours:
            cnt_perimeter = cv2.arcLength(cnt, True)
            cnt_approximation = cv2.approxPolyDP(cnt, 0.04 * cnt_perimeter, True)

            if (len(cnt_approximation) == 4
                    and abs(cv2.contourArea(cnt_approximation) > self.min_area)
                    and cv2.isContourConvex(cnt_approximation)):
                max_cosine = 0
                for i in range(2, 5):
                    cosine = abs(find_cosine(cnt_approximation[i % 4],
                                             cnt_approximation[i - 2],
                                             cnt_approximation[i - 1]))
                    max_cosine = max(max_cosine, cosine)

                if max_cosine < self.cosine_limit:
                    squares.append(cnt_approximation)
        return squares
