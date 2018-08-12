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
import queue
import process_img
import threshold
import model


center_distances = queue.Queue(maxsize=15)
distanceSum = 0


def get_direction(center_dist):
    direction = ''
    if center_dist > 0:
        direction = 'right'
    elif center_dist < 0:
        direction = 'left'
    return direction


def get_car_position(l_fit, r_fit, w, h):
    xm_per_pix = 3.7 / 700
    center_dist = 0
    lane_center_position = 0
    if r_fit is not None and l_fit is not None:
        car_position = w / 2
        l_fit_x_int = l_fit[0] * h ** 2 + l_fit[1] * h + l_fit[2]
        r_fit_x_int = r_fit[0] * h ** 2 + r_fit[1] * h + r_fit[2]
        lane_center_position = (r_fit_x_int + l_fit_x_int) / 2
        center_dist = (car_position - lane_center_position) * xm_per_pix

    global distanceSum
    if (center_distances.full()):
        el = center_distances.get()
        distanceSum -= el

    center_distances.put(center_dist)
    distanceSum += center_dist

    no_of_distance_values = center_distances.qsize()
    center_dist = distanceSum / no_of_distance_values
    return center_dist, lane_center_position

width_lane_avg=[]
radius_values = queue.Queue(maxsize=15)
radius_sum=0


def calc_radius_position(combined, l_fit, r_fit, l_lane_inds, r_lane_inds, lane_width):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
    left_curverad, right_curverad, center_dist, width_lane = (0, 0, 0, 0)
    h = combined.shape[0]
    w = combined.shape[1]
    ploty = np.linspace(0, h - 1, h)
    y_eval = np.max(ploty)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = combined.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Extract left and right line pixel positions
    leftx = nonzerox[l_lane_inds]
    lefty = nonzeroy[l_lane_inds]
    rightx = nonzerox[r_lane_inds]
    righty = nonzeroy[r_lane_inds]

    if len(leftx) != 0 and len(rightx) != 0:
        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)

        # applying the formula for
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * left_fit_cr[0])
        right_curverad = ((1 + (
                    2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * right_fit_cr[0])

        width_lane = lane_width * xm_per_pix
        if (len(width_lane_avg) != 0):
            avg_width = (sum(width_lane_avg) / len(width_lane_avg))
            if abs(avg_width - width_lane) < 0.5:
                width_lane_avg.append(width_lane)
            else:
                width_lane = avg_width

    # Averaging radius value over past 15 frames
    global radius_sum
    if (radius_values.full()):
        el = radius_values.get()

        radius_sum -= el
    curve_radius = (left_curverad + right_curverad) / 2
    radius_values.put(curve_radius)
    radius_sum += curve_radius

    no_of_radius_values = radius_values.qsize()
    curve_radius = radius_sum / no_of_radius_values
    #     print(curve_radius, radius_sum,no_of_radius_values)

    center_dist, lane_center_position = get_car_position(l_fit, r_fit, w, h)  # getting the car distance from the center
    return curve_radius, center_dist, width_lane, lane_center_position

