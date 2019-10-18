#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pyrealsense2 as rs
import matplotlib as plt
import cv2
import logging
import numpy as np
import time


# In[2]:


ctx = rs.context()
devices = ctx.query_devices()


# In[3]:


devices


# In[4]:


var = [i.get_info(rs.camera_info.serial_number) for i in devices]


# In[5]:


var


# In[10]:


#cam1: configure depth and color streams 
pipeline1 = rs.pipeline()
config1 = rs.config()
config1.enable_device(var[0])
config1.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 90)
config1.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
config1.enable_record_to_file('multicam1.bag')
print('cam1 enabled')


# In[11]:


#cam1: configure depth and color streams 
pipeline2 = rs.pipeline()
config2 = rs.config()
config2.enable_device(var[1])
config2.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 90)
config2.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
config2.enable_record_to_file('multicam2.bag')
print('cam2 enabled')


# In[12]:


#start stream
pipeline1.start(config1)
pipeline2.start(config2)
print('cams configured')


# In[13]:


try:
    start = time.time()
    while time.time() - start <10:
        
        #camera 1
        frames1 = pipeline1.wait_for_frames()
        depthframe1 = frames1.get_depth_frame()
        colframe1 = frames1.get_color_frame()
        if not depthframe1 or not colframe1:
            continue

        #convert images to numpy arrays
        depthim1 = np.asanyarray(depthframe1.get_data())
        colim1 = np.asanyarray(colframe1.get_data())
        
        #apply color map on depth image, convert to 8bit pixel
        depthcolmap1 = cv2.applyColorMap(cv2.convertScaleAbs(depthim1, alpha=0.5), cv2.COLORMAP_JET)
        
        #camera 2
        frames2 = pipeline2.wait_for_frames()
        depthframe2 = frames2.get_depth_frame()
        colframe2 = frames2.get_color_frame()
        if not depthframe2 or not colframe2:
            continue

        #convert images to numpy arrays
        depthim2 = np.asanyarray(depthframe2.get_data())
        colim2 = np.asanyarray(colframe2.get_data())
        
        #apply color map on depth image, convert to 8bit pixel
        depthcolmap2 = cv2.applyColorMap(cv2.convertScaleAbs(depthim2, alpha=0.5), cv2.COLORMAP_JET)
        
        
        #stack all images horizontally
        images = np.hstack((colim1, depthcolmap1, colim2, depthcolmap2))
        
        #show images from cameras
        cv2.namedWindow('RealSense', cv2.WINDOW_NORMAL)
        cv2.imshow('RealSense', images)
        cv2.waitKey(1)
    
finally:
    #stop streaming
    pipeline1.stop()
    pipeline2.stop()
    cv2.destroyAllWindows()
    del pipeline1
    del config1
    del pipeline2
    del config2


# In[ ]:




