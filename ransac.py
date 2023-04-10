import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import open3d as o3d
from sklearn.neighbors import KDTree

from matplotlib import cm
from matplotlib.ticker import LinearLocator

from typing import List, Tuple, Callable

np.random.seed(0)

NEAREST_NEIGHBOR_COUNT = 12

  
def another_draw_planes(segments, max_plane_idx):
  import pyvista as pv
  import numpy as np

  # Create random points
  # points = np.random.rand(1000, 3)

  for i in range(max_plane_idx):
    
    points = np.array(segments[i].points)


    # Create a mesh from the points
    mesh = pv.PolyData(points)

    # Create a surface from the mesh
    surface = mesh.delaunay_3d()

    # Plot the surface
    surface.plot(show_edges=True)


def draw_planes_o3d(segments, max_plane_idx):
  # Generate random points
  # points = np.random.rand(1000, 3)

  # Compute convex hull

  hulls = []
  for i in range(max_plane_idx):
    hull = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
        o3d.geometry.PointCloud(segments[i]), alpha=0.1)
    
    hulls.append(hull)

  # Draw plane
  o3d.visualization.draw_geometries(hulls)



def determine_thresold(xyz):
  xyz = np.array(xyz)

  
  tree = KDTree(np.array(xyz), leaf_size=2)
  nearest_dist, nearest_ind = tree.query(xyz, k=8)
  mean_distance = np.mean(nearest_dist[:,1:])

  return mean_distance



# https://github.com/salykovaa/ransac/blob/main/fit_plane.py
def ransac_plane(xyz, threshold=0.01, iterations=1000):
  
  xyz = np.array(xyz)

  inliers=[]
  n_points=len(xyz)

  i=1

  while i<iterations:
    idx_samples = np.random.choice(range(n_points), 3)

    pts = xyz[idx_samples]

    vecA = pts[1] - pts[0]
    vecB = pts[2] - pts[0]
    normal = np.cross(vecA, vecB)
    # a,b,c = normal / np.linalg.norm(normal)
    a, b, c = normal
    d=-np.sum(normal*pts[1])

    distance = (a * xyz[:,0] + b * xyz[:,1] + c * xyz[:,2] + d) / np.sqrt(a ** 2 + b ** 2 + c ** 2) 

    idx_candidates = np.where(np.abs(distance) <= threshold)[0]

    if len(idx_candidates) > len(inliers):
      equation = [a,b,c,d]
      inliers = idx_candidates
      best_plane_point_indices = idx_samples
    
    i+=1
  return equation, inliers, best_plane_point_indices
