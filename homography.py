import cv2
import numpy as np

MIN_MATCH_COUNT = 10

# Query image
img = cv2.imread("images/kariusbaktus_v2.png", cv2.IMREAD_GRAYSCALE)

cap = cv2.VideoCapture(0)

# Features SIFT or ORB (Orb doesn't work with flann?)
sift = cv2.xfeatures2d.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(img, None)
#orb = cv2.ORB_create()
#keypoints, descriptors = orb.detectAndCompute(img, None)
#img = cv2.drawKeypoints(img, keypoints, img)

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict()

# Feature matching
flann = cv2.FlannBasedMatcher(index_params, search_params)

while True:
    # Train image
    _, frame = cap.read()
    grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    kp_grayframe, desc_grayframe = sift.detectAndCompute(grayframe, None)
    #grayframe = cv2.drawKeypoints(grayframe, kp_grayframe, grayframe)
    matches = flann.knnMatch(descriptors, desc_grayframe, k=2)

    '''
    # Alternative method
    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]

    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            matchesMask[i] = [1, 0]

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=0)
                       
    img3 = cv2.drawMatchesKnn(img, keypoints, grayframe, kp_grayframe, matches, None, **draw_params)
    '''

    good = []
    for m, n in matches:
        if m.distance < 0.6*n.distance:
            good.append(m)

    # Homography
    if len(good) > MIN_MATCH_COUNT:
        query_pts = np.float32([keypoints[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        train_pts = np.float32([kp_grayframe[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        # Perspective transform
        h, w = img.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, matrix)

        homography = cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3, cv2.LINE_AA)
        cv2.imshow("Homography", homography)
    else:
        cv2.imshow("Homography", grayframe)
    #img3 = cv2.drawMatches(img, keypoints, grayframe, kp_grayframe, good, grayframe)

    #cv2.imshow("Matches", img3)
    #cv2.imshow("Image", img)
    #cv2.imshow("Frame", grayframe)

    key = cv2.waitKey(1)
    # Supposed to be "s" key
    if key == 27:
        break

cv2.release()
cv2.destroyAllWindows()