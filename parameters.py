from enum import Enum

import pprint

#create paths and load data


input_paths = {
    'mug': "dataset/03797390/fad118b32085f3f2c2c72e575af174cd_8x8.npz",
    'aeroplane': "dataset/02691156/1a04e3eab45ca15dd86060f189eb133_8x8.npz",
    'table':  'dataset/04379243/1a2914169a3a1536a71646339441ab0c_8x8.npz',
    'skateboard': 'dataset/04225987/1ad8d7573dc57ebc247a956ed3779e34_8x8.npz',
    'laptop': 'dataset/03642806/1a46d6683450f2dd46c0b76a60ee4644_8x8.npz'
}

parameters = {
    # not changing
    'cluster_distance_cutoff': 2, # used if automatic_num_planes is True
    'nearest_neighbor_count': 12,
    'iterations': 1000,
    'noise_amount': 0.01,
    'threshold_inside_loop': False, # False is better for noisy
    'top_points_to_take': 1, # or 1, or 0.2


    # changing
    'input_path': input_paths['aeroplane'],
    'object_name': 'aeroplane',
    'add_noise': False,
    'sampling_method': 'random', # 'random', 'convex_hull' or 'refined_convex_hull',
    'automatic_num_planes': False, # True doesnt work for noise,
    'max_plane_idx': 4, # this is the number of planes if automatic_num_planes is False. if automatic_num_planes is True, this is the maximum number of planes to try
}

def generate_parameters_to_experiment():
    # cloning above parameter list for different objects

    #copy an element multiple times in python array
    cloned_parameters = [parameters.copy() for _ in range(len(input_paths))]

    parameters_to_experiment = []





    for i, path in enumerate(input_paths):
        print(i, path)

        changed_parameters = cloned_parameters[i]

        changed_parameters['input_path'] = input_paths[path]
        changed_parameters['object_name'] = str(path)
        parameters_to_experiment.append(changed_parameters)

    


    return parameters_to_experiment

