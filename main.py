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
ap.add_argument("-s", "--shape", type=str,
                default="8,5",
                help="Shape of the board, two numbers separated by comma")
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
    board_shape = [int(i) for i in args["shape"].split(',')]
except:
    print("Invalid shape '{}'. Please use 2 values separated by comma. For example: 8,5".format(args["shape"]))
    sys.exit(0)
else:
    if len(board_shape) != 2:
        print("Invalid shape '{}'. Please use 2 values separated by comma. For example: 8,5".format(args["shape"]))
        sys.exit(0)


# verify that the supplied ArUCo tag exists and is supported by OpenCV
if ARUCO_DICT.get(args["type"], None) is None:
    print("[INFO] ArUCo tag of '{}' is not supported".format(args["type"]))
    sys.exit(0)

# load the ArUCo dictionary and grab the ArUCo parameters
print("[INFO] detecting '{}' tags...".format(args["type"]))
arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[args["type"]])
arucoParams = cv2.aruco.DetectorParameters_create()

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)


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


def findIntersection(p1, p2, p3, p4):
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = p1, p2, p3, p4
    if (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4):
        px = ((x1*y2-y1*x2)*(x3-x4) - (x1-x2)*(x3*y4-y3*x4)) / ((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4))
        py = ((x1*y2-y1*x2)*(y3-y4) - (y1-y2)*(x3*y4-y3*x4)) / ((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4))
        return [px, py]
    else:
        print("lines do not intercept!")


def find_aruco_corners(corners, ids, b_shape):
    ids_flat = ids.flatten()
    corner_markers = []
    m, n = b_shape
    top_ids = [i for i in range(0, m, 1) if i in ids_flat]
    bot_ids = [i for i in range(m * (n - 1), n * m, 1) if i in ids_flat]
    left_ids = [i for i in range(0, m * n, m) if i in ids_flat]
    right_ids = [i for i in range(m - 1, n * m, m) if i in ids_flat]

    try:
        # topLeft and topRight points of 0, and -1 elements in top ids
        top_line = np.squeeze(corners[np.squeeze(np.where(ids_flat == top_ids[0]))])[0], \
                   np.squeeze(corners[np.squeeze(np.where(ids_flat == top_ids[-1]))])[1]
        # bottomLeft and bottomRight points of 0, and -1 elements in bot ids
        bot_line = np.squeeze(corners[np.squeeze(np.where(ids_flat == bot_ids[0]))])[3], \
                   np.squeeze(corners[np.squeeze(np.where(ids_flat == bot_ids[-1]))])[2]
        # topRight and bottomRight points of 0, and -1 elements in right ids
        right_line = np.squeeze(corners[np.squeeze(np.where(ids_flat == right_ids[0]))])[1], \
                     np.squeeze(corners[np.squeeze(np.where(ids_flat == right_ids[-1]))])[2]
        # bottomLeft and topLeft points of 0, and -1 elements in left ids
        left_line = np.squeeze(corners[np.squeeze(np.where(ids_flat == left_ids[0]))])[0], \
                    np.squeeze(corners[np.squeeze(np.where(ids_flat == left_ids[-1]))])[3]

    except Exception as e:
        print(e)

    else:
        if 0 in ids_flat:
            corns_tl = np.squeeze(corners[np.squeeze(np.where(ids_flat == 0))])
            (topLeft_tl, topRight_tl, bottomRight_tl, bottomLeft_tl) = corns_tl
            topLeft = topLeft_tl
        else:
            topLeft = findIntersection(*top_line, *left_line)

        if m*n-1 in ids_flat:
            corns_br = np.squeeze(corners[np.squeeze(np.where(ids_flat == m*n-1))])
            (topLeft_br, topRight_br, bottomRight_br, bottomLeft_br) = corns_br
            bottomRight = bottomRight_br
        else:
            bottomRight = findIntersection(*bot_line, *right_line)

        if m-1 in ids_flat:
            corns_tr = np.squeeze(corners[np.squeeze(np.where(ids_flat == m-1))])
            (topLeft_tr, topRight_tr, bottomRight_tr, bottomLeft_tr) = corns_tr
            topRight = topRight_tr
        else:
            topRight = findIntersection(*top_line, *right_line)

        if m*(n-1) in ids_flat:
            corns_bl = np.squeeze(corners[np.squeeze(np.where(ids_flat == m*(n-1)))])
            (topLeft_bl, topRight_bl, bottomRight_bl, bottomLeft_bl) = corns_bl
            bottomLeft = bottomLeft_bl
        else:
            bottomLeft = findIntersection(*bot_line, *left_line)

        corner_markers = (topLeft, topRight, bottomRight, bottomLeft)

    return corner_markers


# loop over the frames from the video stream
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=1000)
    # detect ArUco markers in the input frame
    (corners, ids, rejected) = cv2.aruco.detectMarkers(frame, arucoDict, parameters=arucoParams)

    if len(corners) > 1:
        board_corners = find_aruco_corners(corners, ids, board_shape)
        if board_corners:
            frame = replace_with_image(frame, board_corners, image)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # break from the loop
    if key in [ord("q"), 27]:
        break

cv2.destroyAllWindows()
vs.stop()
