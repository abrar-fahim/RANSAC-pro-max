import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
import scipy.spatial as sp

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

def sample_convex_hull(points, n):
  # Compute the convex hull of the points
  hull = sp.ConvexHull(points)
  # Get the indices of the vertices of the hull
  vertices = hull.vertices
  #print('vertices', vertices.shape)
  # Randomly sample n indices from the vertices
  sample_indices = np.random.choice(vertices, size=n, replace=False)
  # Return the sampled points
  return points[sample_indices]

# https://github.com/salykovaa/ransac/blob/main/fit_plane.py
def ransac_plane(xyz, threshold=0.01, iterations=1000):
  
  xyz = np.array(xyz)

  inliers=[]
  n_points=len(xyz)

  inlier_counts = []
  iteration_number = []

  i=1

  while i<iterations:
    pts = sample_convex_hull(xyz, 3)

    vecA = pts[1] - pts[0]
    vecB = pts[2] - pts[0]
    normal = np.cross(vecA, vecB)
    a, b, c = normal
    d=-np.sum(normal*pts[1])

    distance = (a * xyz[:,0] + b * xyz[:,1] + c * xyz[:,2] + d) / np.sqrt(a ** 2 + b ** 2 + c ** 2) 

    idx_candidates = np.where(np.abs(distance) <= threshold)[0]

    
    if len(idx_candidates) > len(inliers):
      inlier_counts.append(len(idx_candidates))
      iteration_number.append(i)

      
      equation = [a,b,c,d]
      inliers = idx_candidates
    
    i+=1

  print('inliers', len(inliers))
  print('iteration number', iteration_number[-1])
  plt.scatter(iteration_number, inlier_counts)
  plt.show()
  return equation, inliers
