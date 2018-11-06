import cv2
import numpy as np

img1 = cv2.imread("images/reach_eagle.jpg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("images/me_reach_eagle.jpg", cv2.IMREAD_GRAYSCALE)



# ORB detector
orb = cv2.ORB_create()
keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

# Brute force matching
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)
matches = sorted(matches, key=lambda x:x.distance)

matching_results = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:10], None, flags=2)

cv2.imshow("img1", img1)
cv2.imshow("img2", img2)
cv2.imshow("Matching results", matching_results)
cv2.waitKey(0)
cv2.destroyAllWindows()

