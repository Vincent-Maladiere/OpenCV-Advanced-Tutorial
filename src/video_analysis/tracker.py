import cv2
import argparse
import time

OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.TrackerCSRT_create,
    "rpn": cv2.TrackerDaSiamRPN_create,
    "go_turn": cv2.TrackerGOTURN_create,
    "kcf": cv2.TrackerKCF_create,
    "mil": cv2.TrackerMIL_create,
    #"tld": cv2.TrackerTLD_create,
    #"medianflow": cv2.TrackerMedianFlow_create,
    #"mosse": cv2.TrackerMOSSE_create,
    #"boosting": cv2.TrackerBoosting_create,
}


def main():

    parser = argparse.ArgumentParser(description="tracker")
    parser.add_argument("--algorithm", help="", default="csrt", type=str)
    parser.add_argument("--input_video", help="", default="", type=str)
    parser.add_argument("--camera_id", help="", default=0, type=int)
    args = parser.parse_args()
    print(args)

    if args.input_video:
        cam = cv2.VideoCapture(args.input_video)
    else:
        cam = cv2.VideoCapture(args.camera_id)

    time.sleep(3)

    ret, frame = cam.read()

    bbox = cv2.selectROI(frame)

    tracker = OPENCV_OBJECT_TRACKERS[args.algorithm]()
    ret = tracker.init(frame, bbox)

    while True:

        ret, frame = cam.read()
        if not ret:
            break

        ret, bbox = tracker.update(frame)

        if ret:
            (x, y, w, h) = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2, 1)
        else:
            cv2.putText(
                frame, "Error", (100, 0), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
            )

        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
