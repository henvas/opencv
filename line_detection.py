import cv2
import numpy as np

video = cv2.VideoCapture("images/test_countryroad.mp4")

while True:
    ret, frame = video.read()
    if not ret:
        video = cv2.VideoCapture("images/test_countryroad.mp4")
        continue

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # mask for yellow color
    mask1 = cv2.inRange(hsv, (14, 90, 150), (25, 255, 255))

    # mask for white color
    mask2 = cv2.inRange(hsv, (0, 0, 230), (255, 50, 255))
    #lower_white = np.array([0, 0, 255 - sensitivity])
    #upper_white = np.array([255, sensitivity, 255])

    # final mask
    mask = cv2.bitwise_or(mask1, mask2)

    # applied Gaussian Blur to make the edges smoother
    kernel = np.ones((5, 5), np.float32) / 25
    mask = cv2.filter2D(mask, -1, kernel)

    # find edges
    edges = cv2.Canny(mask, 75, 150)

    # find lines with hough transform w/ROI = [650:1080, 0:1920]
    lines = cv2.HoughLinesP(edges[650:1080, 0:1920], 1, np.pi / 180, 50, maxLineGap=50)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame[650:1080, 0:1920], (x1, y1), (x2, y2), (0, 255, 0), 5)



    cv2.imshow("frame", frame)
    cv2.imshow("mask", edges)

    key = cv2.waitKey(25)
    if key == ord('a'):
        break

video.release()
cv2.destroyAllWindows()
