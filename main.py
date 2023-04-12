#libraries used
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from ransac import *

from parameters import parameters

NOISE_TOLERANCE_FACTOR = 2



with np.load(parameters['input_path']) as data:
    pcdnp = data['pc']

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pcdnp)


segment_models={}
segments={}
max_plane_idx= parameters['max_plane_idx']
rest=pcd


if parameters['add_noise']:
    # add noise to rest points
    rest = add_noise(rest)


# get threshold
start_threshold = determine_thresold(rest.points)

# print("noise ratio", get_noise_ratio_using_dbscan(rest.points, start_threshold))

# cluster_distance_cutoff = int(2 / (1 - get_noise_ratio_using_dbscan(rest.points, start_threshold)))

num_planes = 0
for i in range(max_plane_idx):
    colors = plt.get_cmap("tab20")(i)

    if parameters['threshold_inside_loop']:
        threshold = determine_thresold(rest.points)
    else:
        threshold = start_threshold

    if parameters['automatic_num_planes']:

        # if 0, then 2
        # if 1, then inf

        # print("cluster_distance_cutoff", cluster_distance_cutoff)

        calculated_threshold = determine_thresold(rest.points)
        print('calculated_threshold', calculated_threshold)
        if calculated_threshold > start_threshold * parameters['cluster_distance_cutoff']:
        # if calculated_threshold > start_threshold * cluster_distance_cutoff:
            break

    
    num_planes += 1

    segment_models[i], inliers  = ransac_plane(rest.points, threshold=threshold, iterations=parameters['iterations'], sampling_method=parameters['sampling_method'])

    segments[i]=rest.select_by_index(inliers)

    # draw plane in open3d
    segments[i].paint_uniform_color(list(colors[:3]))
    rest = rest.select_by_index(inliers, invert=True)
    print("pass",i,"/",max_plane_idx,"done.")

#draw_planes(segments, num_planes)


o3d.visualization.draw_geometries([segments[i] for i in range(num_planes)])
#o3d.visualization.draw_geometries([segments[i] for i in range(num_planes)]+[rest])