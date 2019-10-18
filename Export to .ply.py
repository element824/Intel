#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pyrealsense2 as rs
import numpy as np
import open3d as o3d
from pyntcloud import PyntCloud
from plyfile import PlyData, PlyElement


# In[5]:


plydata = PlyData.read('/home/alissa/IntelRealSense/bunny/data/bun000.ply')


# In[9]:


plydata


# In[4]:


#declare realsense pipeline
pipe = rs.pipeline()
config = rs.config()
rs.config.enable_device_from_file(config,'multicam1.bag')
config.enable_stream(rs.stream.depth,848,480, rs.format.z16, 90)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
pipe.start(config)
print('file enabled')


# In[10]:


profile = pipe.get_active_profile()
depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
depth_intrinsics = depth_profile.get_intrinsics()
w,h = depth_intrinsics.width, depth_intrinsics.height
print('got stream profile and camera intrinsics')

#declare pointcloud object for calculating 
#pointclouds and texture mappings
pc = rs.pointcloud()
decimate = rs.decimation_filter()

colorizer = rs.colorizer()
print('processed blocks')
points = rs.points()


# In[12]:


try:
    while True:
        frames = pipe.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        depth_frame = decimate.process(depth_frame)
        color_frame = frames.get_color_frame()
        depth_intrinsics = rs.video_stream_profile(
            depth_frame.profile).get_intrinsics()

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        depth_colormap = np.asanyarray(colorizer.colorize(depth_frame).get_data())
        mapped_frame, color_source = depth_frame, depth_colormap

        points = pc.calculate(depth_frame)
        pc.map_to(mapped_frame)

        print("Saving to 1.ply...")
        points.export_to_ply("1.ply", color_frame)
    
finally:
    pipe.stop()
    del pipe
    del config

