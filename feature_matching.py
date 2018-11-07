import cv2
import numpy as np

img1 = cv2.imread("images/kariusbaktus.jpg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("images/me_karius.jpg", cv2.IMREAD_GRAYSCALE)



# ORB detector
orb = cv2.ORB_create()
keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

# Brute force matching
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)

matches = sorted(matches, key=lambda x:x.distance)
matching_results = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:50], None, flags=2)
cv2.imshow("Matching results", matching_results)

'''
dmatches = sorted(matches, key=lambda x:x.distance)

## extract the matched keypoints
src_pts = np.float32([keypoints1[m.queryIdx].pt for m in dmatches]).reshape(-1, 1, 2)
dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in dmatches]).reshape(-1, 1, 2)

## find homography matrix and do perspective transform
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
h,w = img1.shape[:2]
pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
dst = cv2.perspectiveTransform(pts,M)

## draw found regions
img2 = cv2.polylines(img2, [np.int32(dst)], True, (0,0,255), 1, cv2.LINE_AA)
cv2.imshow("found", img2)


## draw match lines
res = cv2.drawMatches(img1, keypoints1, img2, keypoints2, dmatches[:20],None,flags=2)

cv2.imshow("orb_match", res);
'''

cv2.waitKey();cv2.destroyAllWindows()

