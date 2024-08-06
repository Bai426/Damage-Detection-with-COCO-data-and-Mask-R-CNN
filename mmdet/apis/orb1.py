import numpy as np
import cv2
from matplotlib import pyplot as plt
from pandas._libs.parsers import k

img1 = cv2.imread('/home/bys2058/SiamMask/data/Sch_Br_800_0801/2020-08-01_160204_800 0001.jpg')          # queryImage
img2 = cv2.imread('/home/bys2058/SiamMask/data/Sch_Br_800_0801/2020-08-01_160204_800 1083.jpg')      # trainImage

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
# Initiate SIFT detector
orb = cv2.ORB_create(nfeatures=1000)
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

index_params = dict(algorithm=6,
                    table_number=6,
                    key_size=12,
                    multi_probe_level=2)
search_params = {}
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)
# matcher = cv2.BFMatcher()
# matches = matcher.knnMatch(des1, des2, k=2)


# # BFMatcher with default params
# bf = cv2.BFMatcher()
# matches = bf.knnMatch(des1,des2, k=2)

# As per Lowe's ratio test to filter good matches
good_matches = []
for m, n in matches:
    if m.distance < 0.5 * n.distance:
        good_matches.append(m)
print('number of kp1', len(kp1))
print('number of kp2', len(kp2))

a_kp1 = []
a_kp2 = []
# for i,(m,n) in enumerate(matches):
#     if m.distance < 0.15*n.distance:
#         # matchesMask[i]=[1,0]
#         a_kp1.append(kp1[i].pt)
#         a_kp2.append(kp2[i].pt)
#         # print(i,kp2[i].pt)
# print('number of a_kp1', len(a_kp1))
# print('number of a_kp2', len(a_kp2))

# Initialize lists
list_kp1 = []
list_kp2 = []
for mat in good_matches:

    # Get the matching keypoints for each of the images
    img1_idx = mat.queryIdx
    img2_idx = mat.trainIdx

    # x - columns
    # y - rows
    # Get the coordinates
    (x1, y1) = kp1[img1_idx].pt
    (x2, y2) = kp2[img2_idx].pt

    # Append to each list
    list_kp1.append((x1, y1))
    list_kp2.append((x2, y2))


b_kp1 = np.array(list_kp1).astype(np.float32)
b_kp2 = np.array(list_kp2).astype(np.float32)
Distance_x = np.mean(b_kp2[0]-b_kp1[0])
Distance_y = np.mean(b_kp2[1]-b_kp1[1])
print('average distance_x', Distance_x)
print('average distance_y', Distance_y)
MIN_MATCHES = 1
if len(good_matches) > MIN_MATCHES:
    src_points = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_points = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    m, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    # corrected_img = cv2.warpPerspective(img1, m, (img2.shape[1], img2.shape[0]))

# Draw first 10 matches.
img3 = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=2)

plt.imshow(img3),plt.show()