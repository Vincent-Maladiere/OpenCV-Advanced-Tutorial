from __future__ import print_function
from __future__ import division
import cv2 as cv
import numpy as np
import argparse


def main():

    parser = argparse.ArgumentParser(description='Code for back projection tutorial in video.')
    parser.add_argument('--input', help='Path to input video.', default="data/demo/cow_video.mp4", type=str)
    parser.add_argument('--ref', help='Path to reference image.', default="data/demo/cow_detail.png", type=str)
    args = parser.parse_args()

    cap = cv.VideoCapture(args.input)
    img_ref = cv.imread(args.ref)

    if cap is None or img_ref is None:
        print('Could not open or find the image:', args.input, args.ref)
        exit(0)

    hist_ref = get_hist_ref(img_ref)

    while True:
        
        ret, frame = cap.read()
        if frame is None:
            break

        frame_back_proj = back_proj(frame, hist_ref)

        cv.imshow("frame", frame)
        cv.imshow("back_proj frame", frame_back_proj)
        
        key = cv.waitKey(30)
        if key == ord('q') or key == 27:
            break


def get_hist_ref(img):

    mask = get_mask(img)

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


if __name__ == "__main__":
    main()