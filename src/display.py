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
import process_img
import threshold
import model
import calculate


def draw_lane(original_img, Combined_img, left_fitx, right_fitx, M):
    new_img = np.copy(original_img)

    warp_zero = np.zeros_like(Combined_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    h, w = Combined_img.shape
    ploty = np.linspace(0, h - 1, num=h)

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(255, 0, 0), thickness=15)
    cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False, color=(255, 0, 0), thickness=15)

    return color_warp, new_img


def Plot_details(laneImage, curv_rad, center_dist, width_lane, lane_center_position):
    offest_top = 0
    copy = np.zeros_like(laneImage)

    h = laneImage.shape[0]
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    text = 'Curve radius: ' + '{:04.2f}'.format(curv_rad) + 'm'
    cv2.putText(laneImage, text, (40, 70 + offest_top), font, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(copy, text, (40, 100 + offest_top), font, 4.0, (255, 255, 255), 3, cv2.LINE_AA)

    abs_center_dist = abs(center_dist)
    direction = calculate.get_direction(center_dist)
    text = '{:04.3f}'.format(abs_center_dist) + 'm ' + direction + ' of center'
    #     cv2.putText(laneImage, 'steering '+direction, (40,110+offest_top), font, 1.5, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(laneImage, '|', (640, 710), font, 2.0, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(laneImage, '|', (int(lane_center_position), 680), font, 2.0, (255, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(laneImage, text, (40, 120 + offest_top), font, 1.5, (255, 255, 255), 2, cv2.LINE_AA)

    text = 'Lane Width: ' + '{:04.2f}'.format(width_lane) + 'm'
    cv2.putText(laneImage, text, (40, 170 + offest_top), font, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(copy, text, (40, 280 + offest_top), font, 4.0, (255, 255, 255), 3, cv2.LINE_AA)

    return laneImage, copy



