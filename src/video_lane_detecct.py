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


video_output1 = 'test1.MOV'
video_input1 = VideoFileClip('project_video.mp4')#.subclip(20,25)
processed_video = video_input1.fl_image(CallPipeline)
# %time processed_video.write_videofile(video_output1, audio=False)
video_input1.reader.close()
video_input1.audio.reader.close_proc()