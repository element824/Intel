#!/usr/bin/env python
# coding: utf-8

# In[43]:


import pyrealsense2 as rs
import cv2
import numpy as np
from math import tan, pi
import matplotlib.pyplot as plt


# In[2]:


ctx = rs.context()
devices = ctx.query_devices()


# In[3]:


var = [i.get_info(rs.camera_info.serial_number) for i in devices]


# In[5]:


var


# In[6]:


def get_extrinsics(src,dst):
    extrinsics = src.get_extrinsics_to(dst)
    R = np.reshape(extrinsics.rotation, [3,3]).T
    T = np.array(extrinsics.translation)
    return (R,T)


# In[7]:


#returns camera matrix from intrinsics
def camera_matrix(intrinsics):
    return np.array([[intrinsics.fx, 0, intrinsics.ppx],
                     [0, intrinsics.fy, intrinsics.ppy],
                    [0,0,1]])


# In[8]:


def callback(frame):
    global frame_data
    if frame.is_frameset():
        frameset = frame.as_frameset()
        f1 = frameset.as_video_frame()
        f2 = frameset.as_video_frame()
        left_data = np.asanyarray(f1.get_data())
        right_data = np.asanyarray(f2.get_data())
        ts = frameset.get_timestap()
        frame_data["left"] = left_data
        frame_data["right"] = right_data
        frame_data["timestamp"] = ts
        frame_mutex.release()


# In[8]:


pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 90)
pipe.start(cfg, callback)


# In[71]:


#depth is stored as one unsigned 16-bit integer per pixel
#distance in meters
#retrieve the depth of pixels in meters
#depth scale - units of the values inside depth frame
depth_sensor = pipe_profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
depth_scale


# In[9]:


dpt_frame = pipe.get_depth_frame().as_depth_frame()
pixel_dist_meters = dpt_frame.get_distance(x,y)
pixel_dist_meters


# In[51]:


profiles.get_streams()


# In[49]:


try:
    #retrieve stream and intrinsic properties
    profiles = pipe.get_active_profile()
    streams = {"left": profiles.get_streams()[0].as_video_stream_profile(),
              "right": profiles.get_streams()[1].as_video_stream_profile()}
    intrinsics = {"left": streams["left"].get_intrinsics(),
                 "right": streams["right"].get_intrinsics()}
    
    
    print("Left camera:", intrinsics["left"])
    print("Right camera:", intrinsics["right"])
    
    K_left = camera_matrix(intrinsics["left"])
    K_right = camera_matrix(intrinsics["right"])
    
    (R,T) = get_extrinsics(streams["left"], streams["right"])
    print("rotation matrix:", R)
    print("translation matrix:", T)

finally:
    pipe.stop()
    del pipe
    del config


# In[ ]:


#transform points from one coordinate space to another after knowing extrinsics
#standard affine transformation using a 3x3 rotation matrix and 3-component translation vector
rs.rs2_transform_point_to_point()


# In[ ]:


#deprojection - 2D pixel location + depth and maps it to 3D point location
rs.rs2_deproject_pixel_to_point()


# In[9]:


#cam1: configure depth and color streams 
pipeline1 = rs.pipeline()
config1 = rs.config()
config1.enable_device(var[0])
config1.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 90)
config1.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
config1.enable_record_to_file('multicam1.bag')
print('cam1 enabled')


# In[10]:


#cam1: configure depth and color streams 
pipeline2 = rs.pipeline()
config2 = rs.config()
config2.enable_device(var[1])
config2.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 90)
config2.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
config2.enable_record_to_file('multicam2.bag')
print('cam2 enabled')


# In[11]:


#start stream
pipeline1.start(config1)
pipeline2.start(config2)
print('cams configured')


# In[12]:


frames1 = pipeline1.wait_for_frames()
depthframe1 = frames1.get_depth_frame()
colframe1 = frames1.get_color_frame()


# In[13]:


frames2 = pipeline2.wait_for_frames()
depthframe2 = frames2.get_depth_frame()
colframe2 = frames2.get_color_frame()


# In[45]:


#get depth info for each pixel from cam1
depth_data = depthframe1.as_frame().get_data()
np.image = np.asanyarray(depth_data)
print(np.image)
print(np.image.shape)
plt.plot(np.image)
plt.show()


# In[37]:


#get depth info for each pixel from cam2
depth_data2 = depthframe2.as_frame().get_data()
np.image2 = np.asanyarray(depth_data2)
print(np.image2)


# In[38]:


sum(np.image2)


# In[30]:


print(sum(np.image).shape)
sum(np.image)


# In[35]:


sum(np.image.T)


# In[96]:


#extrinsics from color frame to depth frame
depth_intrin = depthframe1.profile.as_video_stream_profile().intrinsics
color_intrin = colframe1.profile.as_video_stream_profile().intrinsics
d_to_c_extrin = depthframe1.profile.get_extrinsics_to(colframe1.profile)
d_to_c_extrin


# In[102]:


depthframe1.profile.get_extrinsics_to(depthframe1.profile)


# In[60]:


#depth is stored as one unsigned 16-bit integer per pixel
#distance in meters
#retrieve the depth of pixels in meters
#depth scale - units of the values inside depth frame
depth_sensor = pipe_profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
depth_scale


# In[8]:


#map depth to color
depth_pixel = [240,320] #random point
depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, depth_scale)
color_point = rs.rs2_transform_point_to_point(depth_to_color_extrin, depth_point)
color_pixel = rs.rs2_project_point_to_pixel(color_intrin, color_point)
pipeline.stop()


# In[9]:


print(depth_point)
print(color_point)
print(color_pixel)


# In[ ]:


cv2.calibrateCamera

