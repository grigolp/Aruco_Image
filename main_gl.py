import argparse
import numpy as np
import cv2
import cv2.aruco as aruco
from PIL import Image
from imutils.video import VideoStream
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *


class OpenGLAruco:

    def __init__(self, args):
        self.webcam = VideoStream(src=0).start()
        self.overlay = cv2.imread(args['image'])

        self.directory = aruco.Dictionary_get(ARUCO_DICT[args["type"]])
        self.parameters = aruco.DetectorParameters_create()
        self.board = aruco.GridBoard_create(*board_params, self.directory)

        calibration = cv2.FileStorage(args['calib'], cv2.FILE_STORAGE_READ)
        self.camera_matrix = calibration.getNode("camera_matrix").mat()
        self.dist_coeff = calibration.getNode("dist_coeff").mat()

    def _init_gl(self, Width, Height):
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClearDepth(1.0)
        glDepthFunc(GL_LESS)
        glEnable(GL_DEPTH_TEST)
        glShadeModel(GL_SMOOTH)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(35, 1.3, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

        glLightfv(GL_LIGHT0, GL_POSITION, (-40, 300, 200, 0.0))
        glLightfv(GL_LIGHT0, GL_AMBIENT, (0.2, 0.2, 0.2, 1.0))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.5, 0.5, 0.5, 1.0))
        glEnable(GL_LIGHT0)
        glEnable(GL_LIGHTING)
        glEnable(GL_COLOR_MATERIAL)

        # assign texture
        glEnable(GL_TEXTURE_2D)
        self.texture_webcam = glGenTextures(1)

    def _createTexture(self, image, bind):
        # convert image to OpenGL texture format
        texture = cv2.flip(image, 0)
        texture = Image.fromarray(texture)
        ix = texture.size[0]
        iy = texture.size[1]
        texture = texture.tobytes("raw", "BGRX", 0, -1)

        # create background texture
        glBindTexture(GL_TEXTURE_2D, bind)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexImage2D(GL_TEXTURE_2D, 0, 3, ix, iy, 0, GL_RGBA, GL_UNSIGNED_BYTE, texture)

    def _drawTexture(self, bind):
        glBindTexture(GL_TEXTURE_2D, bind)

        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 1.0)
        glVertex3f(-4.0, -3.0, 0.0)
        glTexCoord2f(1.0, 1.0)
        glVertex3f(4.0, -3.0, 0.0)
        glTexCoord2f(1.0, 0.0)
        glVertex3f(4.0, 3.0, 0.0)
        glTexCoord2f(0.0, 0.0)
        glVertex3f(-4.0, 3.0, 0.0)
        glEnd()

    def _render(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        image = self.webcam.read()
        image = self._handle_aruco(image)
        self._createTexture(image, self.texture_webcam)

        glPushMatrix()
        glTranslatef(0, 0, -10.0)
        # glScale(0.5,0.5,0.5)
        self._drawTexture(self.texture_webcam)
        glPopMatrix()

        self._handle_aruco(image)

        glutSwapBuffers()

    def _handle_aruco(self, frame):
        # (height, width, channels) = image.shape
        m, n, l, s = board_params
        (corners, ids, rejectedImgPoints) = aruco.detectMarkers(frame, self.directory, parameters=self.parameters)

        if ids is not None and corners is not None:
            (corners, ids, rejectedImgPoints, recoveredIds) = aruco.refineDetectedMarkers(frame, self.board, corners,
                                                                                          ids, rejectedImgPoints)
            (ret, rvec, tvec) = aruco.estimatePoseBoard(corners, ids, self.board, self.camera_matrix, self.dist_coeff,
                                                          None, None)
            if ret:
                corner_points = np.float32(
                    [[0, n * (l + s), 0], [m * (l + s), n * (l + s), 0], [m * (l + s), 0, 0], [0, 0, 0]])
                board_corners, jac = cv2.projectPoints(corner_points, rvec, tvec, self.camera_matrix, None)
                board_corners = np.array([tuple(pts.ravel().astype(np.uint32)) for pts in board_corners])
                frame = self._replace_with_image(frame, board_corners, self.overlay.copy())

        return frame

    def _replace_with_image(self, frame, corners, image):
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
            print(e)
        return frame

    def main(self):
        glutInit()
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
        glutInitWindowSize(1500, 1000)
        self.window_id = glutCreateWindow(b"OpenGL AR")

        glutDisplayFunc(self._render)
        glutIdleFunc(self._render)
        self._init_gl(1500, 1000)
        glutMainLoop()

def parse_arguments():
    global ARUCO_DICT, board_params
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--type", type=str,
                    default="DICT_7X7_250",
                    help="type of ArUCo tag to detect")
    ap.add_argument("-i", "--image", type=str,
                    default="./images/mona.jpg",
                    help="Path to image")
    ap.add_argument("-b", "--board", type=str,
                    default="8,5,0.05,0.01",
                    help="Parameters of board: rows, columns, length, separation. Separated by comma")
    ap.add_argument("-c", "--calib", type=str,
                    default="./calibration/calibrationCoefficients.yaml",
                    help="Calibration file path")
    args = vars(ap.parse_args())

    # define names of each possible ArUco tag OpenCV supports
    ARUCO_DICT = {
        "DICT_4X4_50": aruco.DICT_4X4_50,
        "DICT_4X4_100": aruco.DICT_4X4_100,
        "DICT_4X4_250": aruco.DICT_4X4_250,
        "DICT_4X4_1000": aruco.DICT_4X4_1000,
        "DICT_5X5_50": aruco.DICT_5X5_50,
        "DICT_5X5_100": aruco.DICT_5X5_100,
        "DICT_5X5_250": aruco.DICT_5X5_250,
        "DICT_5X5_1000": aruco.DICT_5X5_1000,
        "DICT_6X6_50": aruco.DICT_6X6_50,
        "DICT_6X6_100": aruco.DICT_6X6_100,
        "DICT_6X6_250": aruco.DICT_6X6_250,
        "DICT_6X6_1000": aruco.DICT_6X6_1000,
        "DICT_7X7_50": aruco.DICT_7X7_50,
        "DICT_7X7_100": aruco.DICT_7X7_100,
        "DICT_7X7_250": aruco.DICT_7X7_250,
        "DICT_7X7_1000": aruco.DICT_7X7_1000,
        "DICT_ARUCO_ORIGINAL": aruco.DICT_ARUCO_ORIGINAL,
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
            print("Invalid shape '{}'. Please use 4 values separated by comma. For example: 8,5,0.05,0.01".format(
                args["board"]))
            sys.exit(0)
    except:
        print("Invalid shape '{}'. Please use 4 values separated by comma. For example: 8,5,0.05,0.01".format(
            args["board"]))
        sys.exit(0)

    try:
        calibration = cv2.FileStorage(args["calib"], cv2.FILE_STORAGE_READ)
        camera_matrix = calibration.getNode("camera_matrix").mat()
        dist_coeff = calibration.getNode("dist_coeff").mat()
    except Exception as e:
        print("Can not read file {}".format(args["calib"]))
        print(e)
        sys.exit(0)

    # verify that the supplied ArUCo tag exists and is supported by OpenCV
    if ARUCO_DICT.get(args["type"], None) is None:
        print("[INFO] ArUCo tag of '{}' is not supported".format(args["type"]))
        sys.exit(0)

    return args


if __name__ == '__main__':
    args = parse_arguments()
    openGLAruco = OpenGLAruco(args)
    openGLAruco.main()