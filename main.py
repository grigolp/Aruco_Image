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
            topRight = findIntersection(topRight_0, topLeft_0, topRight_39, bottomRight_39)

        if 32 in ids:
            corns_32 = np.squeeze(corners[np.squeeze(np.where(ids == 32))])
            (topLeft_32, topRight_32, bottomRight_32, bottomLeft_32) = corns_32
            bottomLeft = bottomLeft_32
        else:
            bottomLeft = findIntersection(topLeft_0, bottomLeft_0, bottomRight_39, bottomLeft_39)

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
            topLeft = findIntersection(topRight_7, topLeft_7, bottomLeft_32, topLeft_32)

        if 39 in ids:
            corns_39 = np.squeeze(corners[np.squeeze(np.where(ids == 39))])
            (topLeft_39, topRight_39, bottomRight_39, bottomLeft_39) = corns_39
            bottomRight = bottomRight_39
        else:
            bottomRight = findIntersection(topRight_7, bottomRight_7, bottomRight_32, bottomLeft_32)

        corner_markers = (topLeft, topRight, bottomRight, bottomLeft)


    # topLeft, topRight, bottomRight, bottomLeft = [],[],[],[],

    # if 0 in ids:
    #     corns_0 = np.squeeze(corners[np.squeeze(np.where(ids == 0))])
    #     (topLeft_0, topRight_0, bottomRight_0, bottomLeft_0) = corns_0
    #     topLeft = topLeft_0
    #
    # if 7 in ids:
    #     corns_7 = np.squeeze(corners[np.squeeze(np.where(ids == 7))])
    #     (topLeft_7, topRight_7, bottomRight_7, bottomLeft_7) = corns_7
    #     topRight = topRight_7
    #
    # if 32 in ids:
    #     corns_32 = np.squeeze(corners[np.squeeze(np.where(ids == 32))])
    #     (topLeft_32, topRight_32, bottomRight_32, bottomLeft_32) = corns_32
    #     bottomLeft = bottomLeft_32
    #
    # if 39 in ids:
    #     corns_39 = np.squeeze(corners[np.squeeze(np.where(ids == 39))])
    #     (topLeft_39, topRight_39, bottomRight_39, bottomLeft_39) = corns_39
    #     bottomRight = bottomRight_39
    #
    # try:
    #     if topLeft != []:
    #         topLeft = findIntersection(topRight_7, topLeft_7, bottomLeft_32, topLeft_32)
    #     if not bottomRight!= []:
    #         bottomRight = findIntersection(topRight_7, bottomRight_7, bottomRight_32, bottomLeft_32)
    #     if not topRight!= []:
    #         topRight = findIntersection(topRight_0, topLeft_0, topRight_39, bottomRight_39)
    #     if not bottomLeft!= []:
    #         bottomLeft = findIntersection(topLeft_0, bottomLeft_0, bottomRight_39, bottomLeft_39)
    # except Exception as e:
    #     print('Can not find all corners!')
    #     print(e)
    # else:
    #     corner_markers = (topLeft, topRight, bottomRight, bottomLeft)
    #     print('Corners found!')

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

        for (markerCorner, markerID) in zip(corners, ids):
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners

            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))

            # draw the bounding box of the ArUCo detection
            cv2.line(frame, topLeft, topRight, (0, 255, 0), 2)
            cv2.line(frame, topRight, bottomRight, (0, 255, 0), 2)
            cv2.line(frame, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv2.line(frame, bottomLeft, topLeft, (0, 255, 0), 2)

            # compute and draw the center (x, y)-coordinates of the
            # ArUco marker
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv2.circle(frame, (cX, cY), 4, (0, 0, 255), -1)
            cv2.circle(frame, topLeft, 4, (0, 255, 255), -1)

            # draw the ArUco marker ID on the frame
            cv2.putText(frame, str(markerID),
                        (topLeft[0], topLeft[1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)

            # frame = replace_with_image(frame, markerCorner.reshape((4, 2)), image)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key in [ord("q"), 27]:
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
