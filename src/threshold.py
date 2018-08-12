#! /usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ = "WangXiaolong"
# Date: 2018/8/8

import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import process_img


def canny_img(img):
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)
    xgrad = cv2.Sobel(gray, cv2.CV_16SC1, 1, 0)
    ygrad = cv2.Sobel(gray, cv2.CV_16SC1, 0, 1)
    edge_output = cv2.Canny(xgrad, ygrad, 50, 150)
    canny = cv2.Canny(gray, 100, 200)
    return edge_output, canny


def rgb_channel_converter(img):
    img1 = img[:, :, 0]     # R channel
    img2 = img[:, :, 1]     # G channel
    img3 = img[:, :, 2]     # B channel
    return img1, img2, img3


def hls_channel_converter(img):
    img1 = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)[:, :, 0]  # H channel
    img2 = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)[:, :, 1]  # L channel
    img3 = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)[:, :, 2]  # S channel
    return img1, img2, img3


def luv_channel_converter(img):
    img1 = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)[:, :, 0]  # L channel
    img2 = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)[:, :, 1]  # U channel
    img3 = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)[:, :, 2]  # V channel
    return img1, img2, img3


def lab_channel_converter(img):
    img1 = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)[:, :, 0]  # L channel
    img2 = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)[:, :, 1]  # A channel
    img3 = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)[:, :, 2]  # B channel
    return img1, img2, img3


def ycrcb_channel_converter(img):
    img1 = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)[:, :, 0]  # Y channel
    img2 = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)[:, :, 1]  # Cr channel
    img3 = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)[:, :, 2]  # Cb channel
    return img1, img2, img3


def channel_threshold(image, thresh):
    image = image*(255/np.max(image))
    # 2) Apply a threshold to the L channel
    binary_output = np.zeros_like(image)
    binary_output[(image > thresh[0]) & (image <= thresh[1])] = 1
    return binary_output


def abs_sobel_thresh(img, orient='x',thresh_min=0, thresh_max=255):
    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function and take the absolute value

    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))

    #  Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)

    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    return binary_output

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output


def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    # Return the binary image
    return binary_output


def sobel_image(img, orient='x', thresh_min=0, thresh_max=255, convert=True):
    gray = img
    if (convert):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    sobel = None
    if (orient == 'x'):
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

    sobel_abs = np.absolute(sobel)
    sobel_8bit = np.uint8(255 * sobel_abs / np.max(sobel_abs))
    binary_output = np.zeros_like(sobel_8bit)
    binary_output[(sobel_8bit >= thresh_min) & (thresh_max >= sobel_8bit)] = 1
    return binary_output


def sobel_gradient_image(img, thresh=(0, np.pi / 2), convert=True):
    gray = img
    if (convert):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=15)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=15)

    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)

    grad = np.arctan2(abs_sobely, abs_sobelx)

    binary_output = np.zeros_like(grad)
    binary_output[(grad > thresh[0]) & (grad < thresh[1])] = 1
    return binary_output


def sobel_mag(img, thresh, convert=True):
    gray = img
    if (convert):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

    mag = (sobelx ** 2 + sobely ** 2) ** (0.5)

    sobel_mag_8bit = np.uint8(255 * mag / np.max(mag))
    binary_output = np.zeros_like(sobel_mag_8bit)
    binary_output[(sobel_mag_8bit >= thresh[0]) & (thresh[1] >= sobel_mag_8bit)] = 1

    return binary_output


