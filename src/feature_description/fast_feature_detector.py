import cv2
import numpy as np
from matplotlib import pyplot as plt
import argparse


def fast_algo(thresh):

    fast = cv2.FastFeatureDetector_create(threshold=thresh)
    kp = fast.detect(img, None)
    img2 = cv2.drawKeypoints(img, kp, None, color=(255, 0, 0))

    window_fast = "Fast algo"
    cv2.namedWindow(window_fast)
    cv2.imshow(window_fast, img2)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="threshold for FAST algo")
    parser.add_argument("--input", default="data/demo/cows.jpg")
    args = parser.parse_args()

    img = cv2.imread(args.input)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    source_window = "Source image"
    thresh, max_thresh = 25, 100
    cv2.namedWindow(source_window)
    cv2.createTrackbar("Threshold: ", source_window, thresh, max_thresh, fast_algo)

    cv2.imshow(source_window, img)
    fast_algo(thresh)
    cv2.waitKey()
