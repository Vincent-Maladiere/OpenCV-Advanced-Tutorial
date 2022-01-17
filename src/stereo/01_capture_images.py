import cv2
from pathlib import Path


def main():

    cam_l = cv2.VideoCapture(1)
    cam_r = cv2.VideoCapture(2)

    path_l = "data/left_images"
    path_r = "data/right_images"
    Path(path_l).mkdir(parents=True, exist_ok=True)
    Path(path_r).mkdir(parents=True, exist_ok=True)

    cv2.namedWindow(path_l)
    cv2.namedWindow(path_r)
    idx = 0

    while True:

        ret_l, frame_l = cam_l.read()
        ret_r, frame_r = cam_r.read()
        if not ret_l or not ret_r:
            print("failed to grab frame")
            break

        cv2.imshow(path_l, frame_l)
        cv2.imshow(path_r, frame_r)

        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break

        elif k%256 == 32:
            # SPACE pressed
            img_name_l = f"{path_l}/img{idx}.png"
            img_name_r = f"{path_r}/img{idx}.png"
            cv2.imwrite(img_name_l, frame_l)
            cv2.imwrite(img_name_r, frame_r)
            print("{} written!".format(img_name_l))
            print("{} written!".format(img_name_r))
            idx += 1

    cam_l.release()
    cam_r.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()