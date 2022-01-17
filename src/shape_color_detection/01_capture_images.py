import cv2


def main():

    cam = cv2.VideoCapture(0)

    path = "data/ruban"
    cv2.namedWindow(path)
    idx = 0

    while True:

        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break

        cv2.imshow(path, frame)

        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break

        elif k%256 == 32:
            # SPACE pressed
            img_name = f"{path}/img{idx}.png"
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            idx += 1

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()