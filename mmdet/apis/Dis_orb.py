import numpy as np
import cv2
from matplotlib import pyplot as plt

# importing libraries
import os
import cv2
from PIL import Image
import glob
import re
import csv

img1 = cv2.imread('/home/bys2058/SiamMask/data/Sch_Br_800_0801/2020-08-01_160204_800 0001.jpg')          # queryImage
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# Initiate SIFT detector
orb = cv2.ORB_create(nfeatures=500)
kp1, des1 = orb.detectAndCompute(img1, None)

filenames = glob.glob("/home/bys2058/SiamMask/data/Sch_Br_800_0801/*.jpg")
filenames.sort(key=lambda f: int(re.sub('\D', '', f)))
images = [cv2.imread(img) for img in filenames]

data_file = open('/home/bys2058/SiamMask/data/Sch_Br_800_0801/data/8.csv', 'w', newline='')
# create the csv writer object
csv_writer = csv.writer(data_file)
header1 = ['File_name', 'x_displacement', 'y_displacement']

for i in range(len(images)):
        filename = filenames[i]
        img2 = cv2.imread(filename)      # trainImage
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        kp2, des2 = orb.detectAndCompute(img2, None)

        index_params = dict(algorithm=6,
                            table_number=6,
                            key_size=12,
                            multi_probe_level=2)
        search_params = dict(checks=100)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        # matcher = cv2.BFMatcher()
        # matches = matcher.knnMatch(des1, des2, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.4 * n.distance:
                good_matches.append(m)
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
        Distance_x = np.mean(b_kp2[0] - b_kp1[0])
        Distance_y = np.mean(b_kp2[1] - b_kp1[1])
        csv_writer.writerow([filename, Distance_x, Distance_y])  # thre are some files missing in the csv file
        print(i)
