#! /usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ = "WangXiaolong"
# Date: 2018/8/8

import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import process_img
import threshold
import model


img = cv2.imread('001.jpg')
cut_img = process_img.roi(img)
M, M_inv =process_img.get_m_minv()
trans_img = process_img.transform(cut_img, M)

# combine the color space : combine1
h_hls, l_hls, s_hls = threshold.hls_channel_converter(trans_img)
binary_1_ = threshold.channel_threshold(l_hls, (190,255))         # white line : l_hls(190,255)
l_luv, u_luv, v_luv = threshold.luv_channel_converter(trans_img)
binary_2_ = threshold.channel_threshold(l_luv, (170,255))          # white line : l_luv(170,255)
l_lab, a_lab, b_lab = threshold.lab_channel_converter(trans_img)
binary_3_ = threshold.channel_threshold(l_lab, (190, 255))
y_ycrcb, cr_ycrcb, cb_ycrcb = threshold.ycrcb_channel_converter(trans_img)
binary_4_ = threshold.channel_threshold(y_ycrcb, (190, 255))
r_rgb, g_rgb, b_rgb = threshold.rgb_channel_converter(trans_img)
binary_5_ = threshold.channel_threshold(r_rgb, (180, 255))

combine1_binary = np.zeros_like(binary_1_)
combine1_binary[(binary_5_ == 1)|(binary_1_ == 1)] = 1
cv2.imshow('combine1', combine1_binary)

# combine the different threshold : combine2
binary_1 = threshold.abs_sobel_thresh(trans_img, orient='x',thresh_min=40, thresh_max=255)     # abs_sobel "x" (40,255)
binary_2 = threshold.dir_threshold(trans_img, sobel_kernel=3, thresh=(0.7, 1.1))    # not define
binary_3 = threshold.mag_thresh(trans_img, sobel_kernel=3, mag_thresh=(80, 255))     # mag_thresh (80,255)
binary_4 = threshold.sobel_image(trans_img, orient='x', thresh_min=50, thresh_max=255, convert=True)
binary_5 = threshold.sobel_gradient_image(trans_img, thresh=(1.2, 1.8), convert=True)
binary_6 = threshold.sobel_mag(trans_img, (40, 255), convert=True)

combine2_binary = np.zeros_like(binary_1)
combine2_binary[(binary_1 == 1)|(binary_3 == 1)|(binary_3 == 1)] = 1

combine_binary = np.zeros_like(binary_5_ )
combine_binary[(combine1_binary==1)&(combine2_binary==1)] = 1
# cv2.imshow('combine2', combine2_binary * 255)
# cv2.imshow('combine', combine_binary * 255)

histogram = np.sum(combine_binary[combine_binary.shape[0]//2:,:], axis=0)

np.set_printoptions(threshold=np.inf)       # print all np
print(histogram)                            # print
plt.plot(histogram)
base = np.argmax(histogram[:])
print(base)
print(len(histogram[:]))
left_fit, right_fit, left_lane_inds, right_lane_inds = model.Plot_line(combine_binary, smoothen=False, prevFrameCount=6)
print(left_fit)
cv2.waitKey(0)


