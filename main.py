#libraries used
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from ransac import *

NOISE_TOLERANCE_FACTOR = 2

#create paths and load data
input_path="dataset/02691156/1a04e3eab45ca15dd86060f189eb133_8x8.npz" # aeroplane

# input_path="dataset/02691156/1a32f10b20170883663e90eaf6b4ca52_8x8.npz" # aeroplane 2
# input_path="dataset/03636649/1a5ebc8575a4e5edcc901650bbbbb0b5_8x8.npz" # lamp
# input_path="dataset/03797390/fad118b32085f3f2c2c72e575af174cd_8x8.npz" # microwave
# input_path = 'dataset/03636649/1c6701a695ba1b8228eb8d149efa4062_8x8.npz'
# input_path = 'dataset/04379243/1a2914169a3a1536a71646339441ab0c_8x8.npz'

with np.load(input_path) as data:
    pcdnp = data['pc']

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pcdnp)


segment_models={}

plane_idxs = {}
segments={}
max_plane_idx=4
rest=pcd

# add noise to rest points
# nprest = np.asarray(rest.points)
# min = np.min(nprest, axis=0)
# max = np.max(nprest, axis=0)
# noise = np.random.normal(min*2, max*2, [int(nprest.shape[0]*0.01), 3])
# nprest = np.concatenate((nprest, noise), axis=0)
# rest.points = o3d.utility.Vector3dVector(nprest)

# get threshold
start_threshold = determine_thresold(rest.points)

num_planes = 0
for i in range(max_plane_idx):
    colors = plt.get_cmap("tab20")(i)
    threshold = determine_thresold(rest.points)
    #threshold = start_threshold

    # if threshold > start_threshold * NOISE_TOLERANCE_FACTOR:
    #     break
    num_planes += 1

    segment_models[i], inliers  = ransac_plane(rest.points, threshold=threshold, iterations=1000)

    segments[i]=rest.select_by_index(inliers)

    # draw plane in open3d
    segments[i].paint_uniform_color(list(colors[:3]))
    rest = rest.select_by_index(inliers, invert=True)
    print("pass",i,"/",max_plane_idx,"done.")

#draw_planes(segments, num_planes)


o3d.visualization.draw_geometries([segments[i] for i in range(num_planes)])
#o3d.visualization.draw_geometries([segments[i] for i in range(num_planes)]+[rest])