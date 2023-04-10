import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import open3d as o3d
from sklearn.neighbors import KDTree

from typing import List, Tuple, Callable

np.random.seed(0)

def draw_plane(plane):
  plt.rcParams["figure.figsize"] = [7.00, 3.50]
  plt.rcParams["figure.autolayout"] = True

  x = np.linspace(-10, 10, 100)
  y = np.linspace(-10, 10, 100)

  x, y = np.meshgrid(x, y)
  eq = 0.12 * x + 0.01 * y + 1.09

  fig = plt.figure()

  ax = fig.gca(projection='3d')

  ax.plot_surface(x, y, eq)

  plt.show()



def draw_planes_o3d(segments, max_plane_idx):
  # Generate random points
  # points = np.random.rand(1000, 3)

  # Compute convex hull

  hulls = []
  for i in range(max_plane_idx):
    print('segment ', segments[i])
    hull = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
        o3d.geometry.PointCloud(segments[i]), alpha=0.1)
    
    hulls.append(hull)

  # Draw plane
  o3d.visualization.draw_geometries(hulls)

def draw_plane(plane: List[float], size: float = 1.0) -> None:
    """Draws a plane in 3D.

    Args:
        plane (List[float]): The plane equation in the form ax + by + cz + d = 0.
        size (float, optional): The size of the plane. Defaults to 1.0.
    """
    a, b, c, d = plane

    # Create the plane
    xx, yy = np.meshgrid(range(2), range(2))
    z = (-d - a * xx - b * yy) / c

    # Plot the surface
    ax = plt.axes(projection="3d")
    ax.plot_surface(xx, yy, z, alpha=0.2)

    # Plot the plane
    ax.plot3D(
        [0, size],
        [0, size],
        [(-d - a * size - b * size) / c, (-d - a * size - b * size) / c],
        color="black",
    )

    # Set the limits
    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_zlim(0, size)

    plt.show()

def determine_thresold(xyz):
  xyz = np.array(xyz)

  print('xyz shape ', xyz.shape)
  
  tree = KDTree(np.array(xyz), leaf_size=2)
  nearest_dist, nearest_ind = tree.query(xyz, k=12)
  mean_distance = np.mean(nearest_dist[:,1:])

  return mean_distance



# https://github.com/salykovaa/ransac/blob/main/fit_plane.py
def ransac_plane(xyz, threshold=0.01, iterations=1000):
  
  xyz = np.array(xyz)

  print('xyz shape', xyz.shape)
  inliers=[]
  n_points=len(xyz)
  i=1

  while i<iterations:
    idx_samples = random.sample(range(n_points), 3)
    pts = xyz[idx_samples]

    vecA = pts[1] - pts[0]
    vecB = pts[2] - pts[0]
    normal = np.cross(vecA, vecB)
    a,b,c = normal / np.linalg.norm(normal)
    d=-np.sum(normal*pts[1])

    distance = (a * xyz[:,0] + b * xyz[:,1] + c * xyz[:,2] + d
                ) / np.sqrt(a ** 2 + b ** 2 + c ** 2) 

    idx_candidates = np.where(np.abs(distance) <= threshold)[0]

    if len(idx_candidates) > len(inliers):
      equation = [a,b,c,d]
      inliers = idx_candidates
      best_plane_point_indices = idx_samples
    
    i+=1
  return equation, inliers, best_plane_point_indices



# gpt ransac

import numpy as np

def ransac(data, model, n, k, t, d):
    """
    RANSAC algorithm for robust model fitting.

    Parameters
    ----------
    data : array_like
        Data points to fit model to.
    model : function
        Function that fits a model to a set of data points.
    n : int
        Minimum number of data points required to fit the model.
    k : int
        Maximum number of iterations allowed in the algorithm.
    t : float
        Threshold value for determining inliers.
    d : int
        Minimum number of inliers required to accept a model.

    Returns
    -------
    best_model : array_like
        Best model found by the algorithm.
    best_inliers : array_like
        Inliers for the best model found by the algorithm.
    """
    best_model = None
    best_inliers = None
    best_score = 0

    for i in range(k):
        # Randomly select n data points
        indices = np.random.choice(data.shape[0], n, replace=False)
        sample = data[indices]

        # Fit model to sample data points
        maybe_model = model(sample)

        # Find inliers based on threshold value
        distances = np.abs(model(data) - maybe_model)
        inliers = distances < t

        # Check if this is the best model so far
        score = np.sum(inliers)
        if score > best_score and np.sum(inliers) >= d:
            best_model = maybe_model
            best_inliers = inliers
            best_score = score

    return best_model, best_inliers