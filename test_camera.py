import cv2

from camera import Camera

cam = Camera()

while True:

    frame = cam.read()

    if frame is None:
        break

    cv2.imshow("Camera",frame)

    key = cv2.waitKey(1)

    if key == 27:
        break

cam.release()

cv2.destroyAllWindows()