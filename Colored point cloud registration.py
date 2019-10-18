#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import copy
import open3d as o3d


# In[2]:


def draw_registration_result_original_color(source, target, transformation):
    source_temp = copy.deepcopy(source)
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target])


# In[3]:


print("1. Load two point clouds and show initial pose")
source = o3d.io.read_point_cloud("1.ply")
target = o3d.io.read_point_cloud("2.ply")
# draw initial alignment
current_transformation = np.identity(4)
draw_registration_result_original_color(source, target,
                                        current_transformation)


# In[4]:


# point to plane ICP
current_transformation = np.identity(4)
print("2. Point-to-plane ICP registration is applied on original point")
print("   clouds to refine the alignment. Distance threshold 0.02.")
result_icp = o3d.registration.registration_icp(
        source, target, 0.02, current_transformation,
        o3d.registration.TransformationEstimationPointToPlane())    
print(result_icp)
draw_registration_result_original_color(source, target,
                                        result_icp.transformation)


# In[5]:


voxel_radius = [0.04, 0.02, 0.01]
max_iter = [50, 30, 14]

current_transformation = np.identity(4)
print("3. Colored point cloud registration")
for scale in range(3):
    iter = max_iter[scale]
    radius = voxel_radius[scale]
    print([iter, radius, scale])

    print("3-1. Downsample with a voxel size %.2f" % radius)
    source_down = source.voxel_down_sample(radius)
    target_down = target.voxel_down_sample(radius)

    print("3-2. Estimate normal.")
    source_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
    target_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))

    print("3-3. Applying colored point cloud registration")
    result_icp = o3d.registration.registration_colored_icp(
        source_down, target_down, radius, current_transformation,
        o3d.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                relative_rmse=1e-6,
                                                max_iteration=iter))
    current_transformation = result_icp.transformation
    print(result_icp)
draw_registration_result_original_color(source, target,
                                        result_icp.transformation)


# In[ ]:




