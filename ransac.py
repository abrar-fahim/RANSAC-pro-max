import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
import scipy.spatial as sp

from scipy import stats
import open3d as o3d
from sklearn.cluster import DBSCAN

from parameters import parameters

np.random.seed(0)



def draw_planes(segments, max_plane_idx):
  import pyvista as pv
  import numpy as np

  # Define a list of colors for the planes
  colors = ["red", "green", "blue", "yellow", "orange"]

  # Create a figure and an axis object
  fig = plt.figure()
  ax = fig.add_subplot(projection="3d")

  #print segments
  print("segments", segments)
  for i in range(max_plane_idx):
    points = np.array(segments[i].points)

    X = points[:,0]
    Y = points[:,1]
    Z = points[:,2]
    ax.plot_trisurf(X, Y, Z, color=colors[i])

  plt.show()

def add_noise(rest):
  nprest = np.asarray(rest.points)
  min = np.min(nprest, axis=0)
  max = np.max(nprest, axis=0)
  noise = np.random.normal(min*2, max*2, [int(nprest.shape[0]*parameters['noise_amount']), 3])
  nprest = np.concatenate((nprest, noise), axis=0)
  rest.points = o3d.utility.Vector3dVector(nprest)

  return rest



def determine_thresold(xyz, filtered_points_only=False):
  xyz = np.array(xyz)
  if filtered_points_only:
    xyz = get_non_noisy_points(xyz)
  

  tree = KDTree(np.array(xyz), leaf_size=2)
  nearest_dist, nearest_ind = tree.query(xyz, k=parameters['nearest_neighbor_count']) 
  # nearest dist shape: (n, k)

  nearest_dist_sorted = np.sort(nearest_dist, axis=0)


  top_points_to_take = parameters['top_points_to_take']
  top_nearest_dists = nearest_dist_sorted[:int(nearest_dist_sorted.shape[0]*top_points_to_take),:]

  mean_distance = np.mean(top_nearest_dists[:,1:])
  return mean_distance

def sample_convex_hull(points, n, refined_hull = False):

  
  # Compute the convex hull of the points
  if refined_hull:

    hull = get_refined_hull(points)
  else:
    hull = sp.ConvexHull(points)

  # Get the indices of the vertices of the hull
  vertices = hull.vertices
  # Randomly sample n indices from the vertices
  sample_indices = np.random.choice(vertices, size=n, replace=False)
  # Return the sampled points
  return points[sample_indices]

def get_noise_ratio(points):

  # find outlier points using scikit dbscan


  filtered_points = get_non_noisy_points(points)


  points = np.array(points)
  

  return (points.shape[0] - filtered_points.shape[0]) / points.shape[0]


def get_non_noisy_points(points):
    
    # Compute the initial convex hull
    hull = sp.ConvexHull(points)

    # Compute the pairwise distances between points and hull vertices
    dist = sp.distance_matrix(points, points[hull.vertices])
  
    # Compute the MAD of the distances along each axis
    mad_x = stats.median_abs_deviation(dist[:, 0])
    mad_y = stats.median_abs_deviation(dist[:, 1])
    mad_z = stats.median_abs_deviation(dist[:, 2])
  
    # Filter out noisy points based on MAD thresholds
    threshold_x = np.median(dist[:, 0]) + 3 * mad_x
    threshold_y = np.median(dist[:, 1]) + 3 * mad_y
    threshold_z = np.median(dist[:, 2]) + 3 * mad_z
    mask = (dist[:, 0] < threshold_x) & (dist[:, 1] < threshold_y) & (dist[:, 2] < threshold_z)
    filtered_points = points[mask]
  
    return filtered_points

def get_noise_ratio_using_dbscan(points, threshold):
  

  # Compute the DBSCAN clustering
  clustering = DBSCAN(eps=0.5, min_samples=10).fit(points)
  # clustering = DBSCAN(eps=0.01, min_samples=10).fit(points)

  # Compute the number of noise points
  num_noise = np.count_nonzero(clustering.labels_ == -1)
  # Compute the number of non-noise points
  num_non_noise = len(clustering.labels_) - num_noise

  # Compute the noise ratio
  noise_ratio = num_noise / len(clustering.labels_)

  return noise_ratio


def get_refined_hull(points):

  filtered_points = get_non_noisy_points(points)
  # Compute the refined convex hull
  refined_hull = sp.ConvexHull(filtered_points)

  return refined_hull




# https://github.com/salykovaa/ransac/blob/main/fit_plane.py
def ransac_plane(xyz, threshold=0.01, iterations=1000, sampling_method='random'):
  
  xyz = np.array(xyz)

  inliers=[]
  n_points=len(xyz)

  inlier_counts = []
  iteration_number = []

  i=1

  while i<iterations:

    if sampling_method == 'random':
      idx_samples = np.random.choice(range(n_points), 3, replace=False)
      pts = xyz[idx_samples]

    elif sampling_method == 'convex_hull':
      pts = sample_convex_hull(xyz, 3, refined_hull=False)

    elif sampling_method == 'refined_convex_hull':
       pts = sample_convex_hull(xyz, 3, refined_hull=True)

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
  # plt.scatter(iteration_number, inlier_counts)
  # plt.show()
  return equation, inliers
