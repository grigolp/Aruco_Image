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
print(arucoDict)
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


# loop over the frames from the video stream
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=1000)

    # detect ArUco markers in the input frame
    (corners, ids, rejected) = cv2.aruco.detectMarkers(frame,
                                                       arucoDict, parameters=arucoParams)

    # verify *at least* one ArUco marker was detected
    if len(corners) > 0:
        # flatten the ArUco IDs list
        ids = ids.flatten()

        # loop over the detected ArUCo corners
        for (markerCorner, markerID) in zip(corners, ids):
            # corners = markerCorner.reshape((4, 2))
            # (topLeft, topRight, bottomRight, bottomLeft) = corners
            #
            # topRight = (int(topRight[0]), int(topRight[1]))
            # bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            # bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            # topLeft = (int(topLeft[0]), int(topLeft[1]))
            #
            # # draw the bounding box of the ArUCo detection
            # cv2.line(frame, topLeft, topRight, (0, 255, 0), 2)
            # cv2.line(frame, topRight, bottomRight, (0, 255, 0), 2)
            # cv2.line(frame, bottomRight, bottomLeft, (0, 255, 0), 2)
            # cv2.line(frame, bottomLeft, topLeft, (0, 255, 0), 2)
            #
            # # compute and draw the center (x, y)-coordinates of the
            # # ArUco marker
            # cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            # cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            # cv2.circle(frame, (cX, cY), 4, (0, 0, 255), -1)
            #
            # # draw the ArUco marker ID on the frame
            # cv2.putText(frame, str(markerID),
            #             (topLeft[0], topLeft[1] - 15),
            #             cv2.FONT_HERSHEY_SIMPLEX,
            #             0.5, (0, 255, 0), 2)

            frame = replace_with_image(frame, markerCorner.reshape((4, 2)), image)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
