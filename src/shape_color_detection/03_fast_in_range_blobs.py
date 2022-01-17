import os
import cv2
import json
import argparse
import numpy as np
from pprint import pprint

WINDOW_IN_RANGE = "in_range"
WINDOW_OPENING = "opening"
WINDOW_HOUGH = "hough_line"

CONFIG_FILE = "in_range_hough_line_config.json"


def main():

    parser = argparse.ArgumentParser(description="Code for Hough Line")
    parser.add_argument("--camera", help="Camera ID", default=0, type=int)
    parser.add_argument("--input", help="Input video", default="", type=str)
    args = parser.parse_args()

    if args.input:
        cam = cv2.VideoCapture(args.input)
    else:
        cam = cv2.VideoCapture(args.camera)
    
    config = load_config()

    while cam.isOpened():

        ret, img = cam.read()

        if ret:
            img_thresh = in_range_color(img, config)
            boxes = infer_boxes(img_thresh)
            show_box(img, boxes)            

        else:
            print('no video')
            cam.set(cv2.CAP_PROP_POS_FRAMES, 0)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()


def in_range_color(img, config):

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_thresh = cv2.inRange(
        img_hsv,
        (config["low_H"], config["low_S"], config["low_V"]),
        (config["high_H"], config["high_S"], config["high_V"])
    )
    cv2.imshow(WINDOW_IN_RANGE, img_thresh)

    return img_thresh


def detect_lines(img_thresh, config):

    img_edges = cv2.Canny(img_thresh, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        img_edges,
        1,
        np.pi/360,
        config["min_threshold"],
        np.array([]),
        config["min_line_length"],
        config["max_line_gap"],
    )

    return lines


def infer_boxes(img_thresh):

    img_edges_2 = cv2.Canny(img_thresh, 50, 150)
    contours, _ = cv2.findContours(img_edges_2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for contour in contours:
        contours_poly = cv2.approxPolyDP(contour, 3, True)
        bound_rect = cv2.boundingRect(contours_poly)
        boxes.append(bound_rect)

    return boxes


def show_lines(img, lines):

    if lines is None:
        cv2.imshow(WINDOW_HOUGH, img)

    else:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imshow(WINDOW_HOUGH, img)


def show_box(img, boxes):

    if boxes is None:
        cv2.imshow(WINDOW_HOUGH, img)

    else:
        for box in boxes:
            cv2.rectangle(
                img,
                (box[0], box[1]),
                (box[0]+box[2], box[1]+box[3]),
                (0, 255, 0),
                5,
            )
        cv2.imshow(WINDOW_HOUGH, img)


def load_config():

    path = "src/coupling"
    file_path = os.path.join(path, CONFIG_FILE)

    config = json.load(open(file_path, "r"))

    print("Config loaded")
    pprint(config)

    return config


if __name__ == "__main__":
    main()

