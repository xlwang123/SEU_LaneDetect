#! /usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ = "WangXiaolong"
# Date: 2018/8/11

import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os


def roi(src):
    roi_img = src[560:1260, 650:1750]        # 560:1160
    return roi_img


def get_m_minv():
    src = np.float32([[(60, 600), (430, 350), (610, 350), (1030, 600)]])
    dst = np.float32([[(200, 700), (200, 100), (800, 100), (800, 700)]])
    m = cv2.getPerspectiveTransform(src, dst)
    m_inv = cv2.getPerspectiveTransform(dst, src)
    return m, m_inv


def transform(img, m):
    wrap_img = cv2.warpPerspective(img, m, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    return wrap_img


def reverse_warping(img,M):
    un_warp = cv2.warpPerspective(img, M, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    return un_warp

def gray_edge(img):

    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)
    xgrad = cv2.Sobel(gray, cv2.CV_16SC1, 1, 0)
    ygrad = cv2.Sobel(gray, cv2.CV_16SC1, 0, 1)
    edge_output = cv2.Canny(xgrad, ygrad, 50, 150)
    canny = cv2.Canny(gray, 100, 200)

    return blurred, gray, edge_output, canny




