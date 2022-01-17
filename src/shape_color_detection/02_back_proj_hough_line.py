import cv2
import argparse
import numpy as np

WINDOW_BINARY = "binary_color"
WINDOW_HOUGH = "hough_line"
WINDOW_REF = "reference"

# Hough
global min_threshold
min_threshold = 0

global min_line_length
min_line_length = 0

global max_line_gap
max_line_gap = 20


def main():

    parser = argparse.ArgumentParser(description="Code for Hough Line")
    parser.add_argument("--camera", help="Camera ID", default=0, type=int)
    parser.add_argument("--input", help="Input video", default="", type=str)
    parser.add_argument("--input_roi", help="Input ROI", default="", type=str)

    args = parser.parse_args()

    if args.input:
        cam = cv2.VideoCapture(args.input)
    else:
        cam = cv2.VideoCapture(args.camera)

    cv2.namedWindow(WINDOW_REF)
    cv2.namedWindow(WINDOW_BINARY)
    cv2.namedWindow(WINDOW_HOUGH)

    img_ref = cv2.imread(args.input_roi)
    cv2.imshow(WINDOW_REF, img_ref)

    hist_ref = get_hist_ref(img_ref)

    cv2.createTrackbar("min_threshold", WINDOW_HOUGH, min_threshold, 500, on_threshold_change)
    cv2.createTrackbar("min_line_length", WINDOW_HOUGH, min_line_length, 500, on_min_line_length_change)
    cv2.createTrackbar("max_line_gap", WINDOW_HOUGH, max_line_gap, 500, on_max_line_gap_change)

    while cam.isOpened():

        ret, img = cam.read()
        if ret:
            img_thresh = back_proj(img, hist_ref)
            lines = detect_edges(img_thresh)
            show_lines(img, lines)            
        else:
            print('no video')
            cam.set(cv2.CAP_PROP_POS_FRAMES, 0)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()


def get_hist_ref(img_ref):

    img_hsv = cv2.cvtColor(img_ref, cv2.COLOR_RGB2HSV)

    ch = (0, 0)
    hue = np.empty(img_hsv.shape, img_hsv.dtype)
    cv2.mixChannels([img_hsv], [hue], ch)

    # build hue hist
    bins = 100
    histSize = max(bins, 2)
    ranges = [0, 180] # hue_range

    hist = cv2.calcHist([hue], [0], None, [histSize], ranges, accumulate=False)
    cv2.normalize(hist, hist, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    return hist


def back_proj(frame, hist_ref):

    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

    # build hue
    ch = (0, 0)
    hue = np.zeros(frame_hsv.shape, frame_hsv.dtype)
    cv2.mixChannels([frame_hsv], [hue], ch)

    ranges = [0, 180] # hue_range

    # get backproj
    frame_back_proj = cv2.calcBackProject([hue], [0], hist_ref, ranges, scale=1)

    cv2.imshow(WINDOW_BINARY, frame_back_proj)

    return frame_back_proj


def detect_edges(img_thresh):

    #gray = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_HSV2RGB), 
    edges = cv2.Canny(img_thresh, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi/360,
        min_threshold,
        np.array([]),
        min_line_length,
        max_line_gap,
    )

    return lines


def show_lines(img, lines):

    if lines is None:
        cv2.imshow(WINDOW_HOUGH, img)
    else:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        cv2.imshow(WINDOW_HOUGH, img)


def on_threshold_change(val):
    global min_threshold
    min_threshold = val
    cv2.setTrackbarPos("min_threshold", WINDOW_HOUGH, val)


def on_min_line_length_change(val):
    global min_line_length
    min_line_length = val
    cv2.setTrackbarPos("min_line_length", WINDOW_HOUGH, val)


def on_max_line_gap_change(val):
    global max_line_gap
    max_line_gap = val
    cv2.setTrackbarPos("max_line_gap", WINDOW_HOUGH, val)


if __name__ == "__main__":
    main()

