from enum import Enum

#create paths and load data


# input_path="dataset/02691156/1a32f10b20170883663e90eaf6b4ca52_8x8.npz" # aeroplane 2
# input_path="dataset/03636649/1a5ebc8575a4e5edcc901650bbbbb0b5_8x8.npz" # lamp


# input_path = 'dataset/04379243/1a2914169a3a1536a71646339441ab0c_8x8.npz'

class Sampling_method(Enum):
    CONVEX_HULL = 1
    GREEN = 2
    BLUE = 3

# input paths we're using
# input_path="dataset/03797390/fad118b32085f3f2c2c72e575af174cd_8x8.npz" # mug

# input_path="dataset/02691156/1a04e3eab45ca15dd86060f189eb133_8x8.npz" # aeroplane

input_paths = {
    'mug': "dataset/03797390/fad118b32085f3f2c2c72e575af174cd_8x8.npz",
    'aeroplane': "dataset/02691156/1a04e3eab45ca15dd86060f189eb133_8x8.npz",
    'table':  'dataset/04379243/1a2914169a3a1536a71646339441ab0c_8x8.npz'
}

parameters = {
    # not changing
    'input_path': input_paths['mug'],
    'max_plane_idx': 6,
    'cluster_distance_cutoff': 2, # used if automatic_num_planes is True
    'nearest_neighbor_count': 12,
    'iterations': 1000,
    'automatic_num_planes': False, # True doesnt work for noise
    'noise_amount': 0.01,


    # changing
    'add_noise': False,
    'threshold_inside_loop': False, # False is better for noisy
    'top_points_to_take': 1, # or 1, or 0.2
    'sampling_method': 'random', # 'random', 'convex_hull' or 'refined_convex_hull'
}