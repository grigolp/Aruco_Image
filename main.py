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
                default="DICT_ARUCO_ORIGINAL",
                help="type of ArUCo tag to detect")
ap.add_argument("-i", "--image", type=str,
                default="./Mona Lisa.jpg",
                help="Path to image")
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
    img = image + 1  # replace 0-s with 1-s, used for pasting
    w, h, _ = img.shape
    pts1 = np.float32([[0, 0], [h, 0], [0, w], [h, w]])
    (topLeft, topRight, bottomRight, bottomLeft) = corners

    try:
        perspective = cv2.getPerspectiveTransform(pts1, np.float32([topLeft, topRight, bottomLeft, bottomRight]))
        image_trans = cv2.warpPerspective(img, perspective, frame.shape[:2][::-1])
        idxs_2D = np.sum(image_trans, axis=2) > 0
        idxs_3D = np.repeat(idxs_2D[:, :, np.newaxis], 3, axis=2)
        frame[idxs_3D] = image_trans[idxs_3D]
    except:
        print('Could not paste image!')
    return frame


def findIntersection(p1, p2, p3, p4):
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = p1, p2, p3, p4
    if (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4):
        px= ( (x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4) ) / ( (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4) )
        py= ( (x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4) ) / ( (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4) )
        return [px, py]
    else:
        print("lines do not intercept!")

def find_aruco_corners(corners, ids):
    ids = ids.flatten()

    corner_markers = []

    if 0 in ids and 39 in ids:
        corns_0 = np.squeeze(corners[np.squeeze(np.where(ids == 0))])
        (topLeft_0, topRight_0, bottomRight_0, bottomLeft_0) = corns_0
        corns_39 = np.squeeze(corners[np.squeeze(np.where(ids == 39))])
        (topLeft_39, topRight_39, bottomRight_39, bottomLeft_39) = corns_39

        topLeft = topLeft_0
        bottomRight = bottomRight_39

        if 7 in ids:
            corns_7 = np.squeeze(corners[np.squeeze(np.where(ids == 7))])
            (topLeft_7, topRight_7, bottomRight_7, bottomLeft_7) = corns_7
            topRight = topRight_7
        else:
            id_top = [i for i in [6,5,4,3,2,1,0] if i in ids][0]
            corns_top = np.squeeze(corners[np.squeeze(np.where(ids == id_top))])
            (topLeft_top, topRight_top, bottomRight_top, bottomLeft_top) = corns_top
            id_right = [i for i in [15,23,31,39] if i in ids][0]
            corns_right = np.squeeze(corners[np.squeeze(np.where(ids == id_right))])
            (topLeft_right, topRight_right, bottomRight_right, bottomLeft_right) = corns_right
            topRight = findIntersection(topRight_top, topLeft_top, topRight_right, bottomRight_right)

        if 32 in ids:
            corns_32 = np.squeeze(corners[np.squeeze(np.where(ids == 32))])
            (topLeft_32, topRight_32, bottomRight_32, bottomLeft_32) = corns_32
            bottomLeft = bottomLeft_32
        else:
            id_bot = [i for i in [33,34,35,36,37,38,39] if i in ids][0]
            corns_bot = np.squeeze(corners[np.squeeze(np.where(ids == id_bot))])
            (topLeft_bot, topRight_bot, bottomRight_bot, bottomLeft_bot) = corns_bot
            id_left = [i for i in [24,16,8,0] if i in ids][0]
            corns_left = np.squeeze(corners[np.squeeze(np.where(ids == id_left))])
            (topLeft_left, topRight_left, bottomRight_left, bottomLeft_left) = corns_left
            bottomLeft = findIntersection(topLeft_left, bottomLeft_left, bottomRight_bot, bottomLeft_bot)

        corner_markers = (topLeft, topRight, bottomRight, bottomLeft)


    if 7 in ids and 32 in ids:
        corns_7 = np.squeeze(corners[np.squeeze(np.where(ids == 7))])
        (topLeft_7, topRight_7, bottomRight_7, bottomLeft_7) = corns_7
        corns_32 = np.squeeze(corners[np.squeeze(np.where(ids == 32))])
        (topLeft_32, topRight_32, bottomRight_32, bottomLeft_32) = corns_32

        topRight = topRight_7
        bottomLeft = bottomLeft_32

        if 0 in ids:
            corns_0 = np.squeeze(corners[np.squeeze(np.where(ids == 0))])
            (topLeft_0, topRight_0, bottomRight_0, bottomLeft_0) = corns_0
            topLeft = topLeft_0
        else:
            id_top = [i for i in [1,2,3,4,5,6,7] if i in ids][0]
            corns_top = np.squeeze(corners[np.squeeze(np.where(ids == id_top))])
            (topLeft_top, topRight_top, bottomRight_top, bottomLeft_top) = corns_top
            id_left = [i for i in [8,16,24,32] if i in ids][0]
            corns_left = np.squeeze(corners[np.squeeze(np.where(ids == id_left))])
            (topLeft_left, topRight_left, bottomRight_left, bottomLeft_left) = corns_left
            topLeft = findIntersection(topRight_top, topLeft_top, bottomLeft_left, topLeft_left)

        if 39 in ids:
            corns_39 = np.squeeze(corners[np.squeeze(np.where(ids == 39))])
            (topLeft_39, topRight_39, bottomRight_39, bottomLeft_39) = corns_39
            bottomRight = bottomRight_39
        else:
            id_bot = [i for i in [38,37,36,35,34,33,32] if i in ids][0]
            corns_bot = np.squeeze(corners[np.squeeze(np.where(ids == id_bot))])
            (topLeft_bot, topRight_bot, bottomRight_bot, bottomLeft_bot) = corns_bot
            id_right = [i for i in [31,23,15,7] if i in ids][0]
            corns_right = np.squeeze(corners[np.squeeze(np.where(ids == id_right))])
            (topLeft_right, topRight_right, bottomRight_right, bottomLeft_right) = corns_right
            bottomRight = findIntersection(topRight_right, bottomRight_right, bottomRight_bot, bottomLeft_bot)

        corner_markers = (topLeft, topRight, bottomRight, bottomLeft)
    return corner_markers

# loop over the frames from the video stream
while True:
    frame = vs.read()
    # frame = imutils.resize(frame, width=1000)
    print(frame.shape)
    # detect ArUco markers in the input frame
    (corners, ids, rejected) = cv2.aruco.detectMarkers(frame, arucoDict, parameters=arucoParams)

    if len(corners)>1:
        board_corners = find_aruco_corners(corners, ids)
        if board_corners:
            frame = replace_with_image(frame, board_corners, image)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key in [ord("q"), 27]:
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
