from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2
import sys
import numpy as np


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--type", type=str,
                default="DICT_7X7_250",
                help="type of ArUCo tag to detect")
ap.add_argument("-i", "--image", type=str,
                default="./Mona Lisa.jpg",
                help="Path to image")
ap.add_argument("-b", "--board", type=str,
                default="8,5,0.05,0.01",
                help="Parameters of board: rows, columns, length, separation. Separated by comma")
args = vars(ap.parse_args())

# define names of each possible ArUco tag OpenCV supports
ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
}
# verify that image can be opened by OpenCV
try:
    image = cv2.imread(args["image"])
    image.shape
except:
    print("Image '{}' can not be opened".format(args["image"]))
    sys.exit(0)

try:
    board_params = [int(p) if i < 2 else float(p) for i, p in enumerate(args["board"].split(','))]
    if len(board_params) != 4:
        print("Invalid shape '{}'. Please use 4 values separated by comma. For example: 8,5,0.05,0.01".format(args["board"]))
        sys.exit(0)
except:
    print("Invalid shape '{}'. Please use 4 values separated by comma. For example: 8,5,0.05,0.01".format(args["board"]))
    sys.exit(0)

# verify that the supplied ArUCo tag exists and is supported by OpenCV
if ARUCO_DICT.get(args["type"], None) is None:
    print("[INFO] ArUCo tag of '{}' is not supported".format(args["type"]))
    sys.exit(0)


def replace_with_image(frame, corners, image):
    img = image.astype(np.uint32) + 1  # replace 0-s with 1-s, used for pasting
    img = np.minimum(img, 255).astype(np.uint8)

    w, h, _ = img.shape
    pts1 = np.float32([[0, 0], [h, 0], [0, w], [h, w]])
    (topLeft, topRight, bottomRight, bottomLeft) = corners

    try:
        perspective = cv2.getPerspectiveTransform(pts1, np.float32([topLeft, topRight, bottomLeft, bottomRight]))
        image_trans = cv2.warpPerspective(img, perspective, frame.shape[:2][::-1])
        idxs_3D = image_trans > 0
        frame[idxs_3D] = image_trans[idxs_3D]
    except Exception as e:
        print('Could not paste image!')
        print(e)

    return frame


# load the ArUCo dictionary and grab the ArUCo parameters
print("[INFO] detecting '{}' tags...".format(args["type"]))
arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[args["type"]])
arucoParams = cv2.aruco.DetectorParameters_create()

m, n, l, s = board_params
board = cv2.aruco.GridBoard_create(*board_params, arucoDict)

##########################################################################
camera_matrix = np.array([[1000,  0., 10],
                          [0., 1000, 10],
                          [0., 0., 1.]])

distortions = None
##########################################################################

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=1800)
    # detect ArUco markers in the input frame
    (corners, ids, rejected) = cv2.aruco.detectMarkers(frame, arucoDict, parameters=arucoParams)

    try:
        if ids is not None and corners is not None:
            (corners, ids, rejectedImgPoints, recoveredIds) = cv2.aruco.refineDetectedMarkers(frame, board, corners,
                                                                                              ids, rejected)
            pose, rvec, tvec = cv2.aruco.estimatePoseBoard(corners, ids, board, camera_matrix, distortions, None, None)
            if pose:
                corner_points = np.float32([[0, n*(l+s), 0], [m*(l+s), n*(l+s), 0], [m*(l+s), 0, 0], [0, 0, 0]])
                board_corners, jac = cv2.projectPoints(corner_points, rvec, tvec, camera_matrix, None)
                board_corners = np.array([tuple(pts.ravel().astype(np.uint32)) for pts in board_corners])

                frame = replace_with_image(frame, board_corners, image)
    except Exception as e:
        print(e)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # break from the loop
    if key in [ord("q"), 27]:
        break

cv2.destroyAllWindows()
vs.stop()
