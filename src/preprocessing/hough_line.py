import cv2
import argparse
import numpy as np

NAME_WINDOW = "hough_line"

global min_threshold
min_threshold = 0

global min_line_length
min_line_length = 0

global max_line_gap
max_line_gap = 20


def main():

    parser = argparse.ArgumentParser(description="Code for Hough Line")
    parser.add_argument("--camera", help="Camera ID", default=0, type=int)
    args = parser.parse_args()

    cam = cv2.VideoCapture(args.camera)

    cv2.namedWindow(NAME_WINDOW)
    cv2.createTrackbar("min_threshold", NAME_WINDOW, min_threshold, 500, on_threshold_change)
    cv2.createTrackbar("min_line_length", NAME_WINDOW, min_line_length, 500, on_min_line_length_change)
    cv2.createTrackbar("max_line_gap", NAME_WINDOW, max_line_gap, 500, on_max_line_gap_change)

    while cam.isOpened():

        ret, img = cam.read()
        if ret:
            edges = preprocess(img)
            lines = cv2.HoughLinesP(
                edges,
                1,
                np.pi/360,
                min_threshold,
                np.array([]),
                min_line_length,
                max_line_gap,
            )
            show_lines(img, lines)            
        else:
            print('no video')
            cam.set(cv2.CAP_PROP_POS_FRAMES, 0)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()


def preprocess(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    return edges


def show_lines(img, lines):

    if lines is None:
        cv2.imshow(NAME_WINDOW, img)
    else:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        cv2.imshow(NAME_WINDOW, img)


def on_threshold_change(val):
    global min_threshold
    min_threshold = val
    cv2.setTrackbarPos("min_threshold", NAME_WINDOW, val)


def on_min_line_length_change(val):
    global min_line_length
    min_line_length = val
    cv2.setTrackbarPos("min_line_length", NAME_WINDOW, val)


def on_max_line_gap_change(val):
    global max_line_gap
    max_line_gap = val
    cv2.setTrackbarPos("max_line_gap", NAME_WINDOW, val)



if __name__ == "__main__":
    main()