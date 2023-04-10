import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree

np.random.seed(0)
NEAREST_NEIGHBOR_COUNT = 12
def draw_planes(segments, max_plane_idx):
  import pyvista as pv
  import numpy as np

  # Define a list of colors for the planes
  colors = ["red", "green", "blue", "yellow", "orange"]

  # Create a figure and an axis object
  fig = plt.figure()
  ax = fig.add_subplot(projection="3d")

  for i in range(max_plane_idx):
    
    points = np.array(segments[i].points)

    X = points[:,0]
    Y = points[:,1]
    Z = points[:,2]
    ax.plot_trisurf(X, Y, Z, color=colors[i])

  plt.show()



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
