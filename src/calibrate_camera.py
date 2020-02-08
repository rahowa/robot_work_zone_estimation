import os
import cv2 
import numpy as np 
from glob import glob
import typing as t
from argparse import ArgumentParser


CAM_WIDTH = 640
CAM_HEIGHT = 480
CHESSBOARD_WIDTH = 6
CHESSBOARD_HEIGHT = 9
IMG_SHAPE = (CAM_HEIGHT, CAM_WIDTH)
CHESSBOARD_SHAPE = (CHESSBOARD_HEIGHT, CHESSBOARD_WIDTH)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def take_pictures(num_pictures: int, save_path: str) -> None: 
    cap = cv2.VideoCapture(0)
    cap.set(3, CAM_HEIGHT)
    cap.set(4, CAM_WIDTH)
    num_taken_pictures = 0
    
    while num_taken_pictures < num_pictures:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners_founded, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SHAPE, None)
        if corners_founded:
            corners        = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            good_chess_img = cv2.drawChessboardCorners(gray, CHESSBOARD_SHAPE, corners, ret)
            good_chess_img = cv2.putText(good_chess_img, "Good view",
                                         (CAM_HEIGHT//2, CAM_WIDTH//2 - 30),
                                         cv2.FONT_HERSHEY_SIMPLEX, 2,
                                         (0, 255, 0))
            cv2.imwrite(
                os.path.join(save_path,
                             f'frame_{num_taken_pictures}.jpg'),
                frame
            )        
            num_taken_pictures += 1
            cv2.imshow("frame", good_chess_img)
        else:
            cv2.imshow("frame", frame)

        if cv2.waitKey(1) % 0xFF == ord('q'):
            break
        
        # elif cv2.waitKey(60) == 32:
            cv2.imwrite(
                os.path.join(save_path,
                             f'frame_{num_taken_pictures}.jpg'),
                frame
            )
            num_taken_pictures += 1
    
    cv2.destroyAllWindows()
    cap.release()



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--tp", action="store_true")
    parser.add_argument("--path", default='./data/calibration', type=str)
    parser.add_argument("--n", default=10, type=int)
    args = parser.parse_args()

    if args.tp:
        take_pictures(args.n, args.path)

    default_obj_points = np.zeros((CHESSBOARD_HEIGHT * CHESSBOARD_WIDTH, 3), np.float32)
    default_obj_points[:, :2] = np.mgrid[0:CHESSBOARD_WIDTH, 0:CHESSBOARD_HEIGHT].T.reshape(-1, 2)

    obj_points = []
    img_points = []

    images = glob(args.path + '/*.jpg')
    for img_path in images:
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SHAPE, None)
        if ret:
            obj_points.append(default_obj_points)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            img_points.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points,
                                                       img_points,
                                                       IMG_SHAPE,
                                                       None, None)
    print(mtx)


# [10097  0       231]
# [0      2495    312]
# [0      0       1  ]