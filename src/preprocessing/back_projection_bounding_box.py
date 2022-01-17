from __future__ import print_function
from __future__ import division
import cv2 as cv
import numpy as np
import argparse

import random as rng
rng.seed(12345)


def main():

    parser = argparse.ArgumentParser(description='Code for back projection tutorial in video.')
    parser.add_argument('--input', help='Path to input image.', default="data/demo/cows.jpg", type=str)
    parser.add_argument('--ref', help='Path to reference image.', default="data/demo/cow_detail.png", type=str)
    parser.add_argument('--use_mask', help='Wether or not using a mask.', default=True, type=bool)
    args = parser.parse_args()

    cap = cv.VideoCapture(args.input)
    img_ref = cv.imread(args.ref)

    if cap is None or img_ref is None:
        print('Could not open or find the image:', args.input, args.ref)
        exit(0)

    hist_ref = get_hist_ref(img_ref, args.use_mask)

    frame = cv.imread(args.input)

    frame_back_proj = back_proj(frame, hist_ref)

    frame_bounding_box, frame_rectangle = bounding_box(frame_back_proj)

    cv.imshow("frame", frame)
    cv.imshow("back_proj frame", frame_back_proj)
    cv.imshow("contours", frame_bounding_box)
    cv.imshow("rectangle", frame_rectangle)
    
    cv.waitKey()


def get_hist_ref(img, use_mask):

    if use_mask:
        mask = get_mask(img)
    else:
        mask = None

    img_hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)

    ch = (0, 0)
    hue = np.empty(img_hsv.shape, img_hsv.dtype)
    cv.mixChannels([img_hsv], [hue], ch)

    # build hue hist
    bins = 100
    histSize = max(bins, 2)
    ranges = [0, 180] # hue_range

    hist = cv.calcHist([hue], [0], mask, [histSize], ranges, accumulate=False)
    cv.normalize(hist, hist, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)

    return hist


def get_mask(img):

    h, w, _ = img.shape
    blank = np.zeros((h, w), dtype=np.uint8)
    mask = cv.circle(blank, center=(w//2, h//2-h//20), radius=w//15, color=255, thickness=-1)

    return mask


def back_proj(frame, hist_ref):

    frame_hsv = cv.cvtColor(frame, cv.COLOR_RGB2HSV)

    # build hue
    ch = (0, 0)
    hue = np.zeros(frame_hsv.shape, frame_hsv.dtype)
    cv.mixChannels([frame_hsv], [hue], ch)

    ranges = [0, 180] # hue_range

    # get backproj
    frame_back_proj = cv.calcBackProject([hue], [0], hist_ref, ranges, scale=1)

    return frame_back_proj


def bounding_box(frame):

    frame = cv.blur(frame, (3,3))

    threshold = 250
    canny_output = cv.Canny(frame, threshold, threshold * 2)
    contours, _ = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    frame_contour = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    frame_rectangle = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)

    n_contour = len(contours)
    contours_poly = [None] * n_contour
    bound_rect = [None] * n_contour
    for idx, contour in enumerate(contours):

        contours_poly[idx] = cv.approxPolyDP(contour, 3, True)
        bound_rect[idx] = cv.boundingRect(contours_poly[idx])

    for idx in range(n_contour):

        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        cv.drawContours(frame_contour, contours_poly, idx, color)
        cv.rectangle(
            frame_rectangle,
            (
                int(bound_rect[idx][0]),
                int(bound_rect[idx][1]),
            ),
            (
                int(bound_rect[idx][0]+bound_rect[idx][2]),
                int(bound_rect[idx][1]+bound_rect[idx][3]),
            ),
            color,
            2,
        )

    return frame_contour, frame_rectangle


if __name__ == "__main__":
    main()