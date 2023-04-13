from parameters import parameters,input_paths

from main import segment_into_planes

import numpy as np

import matplotlib.pyplot as plt
import pprint


# different nn counts

# threshold_list = [4, 12, 30]
# for threshold in threshold_list:
#     np.random.seed(0)
#     parameters['input_path'] = input_paths['mug']
#     parameters['sampling_method'] = 'random'
#     parameters['nearest_neighbor_count'] = threshold
#     graph_random = segment_into_planes(parameters, visualize=True)


# different sampling methods

# threshold_list = [1.5, 2, 3, 4]
# for threshold in threshold_list:
#     np.random.seed(0)
#     parameters['input_path'] = input_paths['aeroplane']
#     parameters['sampling_method'] = 'refined_convex_hull'
#     parameters['cluster_distance_cutoff'] = threshold
#     graph_random = segment_into_planes(parameters, visualize=True)


# adding noise
# parameters['add_noise'] = False
# np.random.seed(0)
# parameters['input_path'] = input_paths['aeroplane']
# parameters['sampling_method'] = 'refined_convex_hull'
# graph_random = segment_into_planes(parameters, visualize=True)



# different sampling methods

threshold_list = [0.01, 0.02, 0.03]
parameters['add_noise'] = True
for threshold in threshold_list:
    np.random.seed(0)
    parameters['noise_amount'] = threshold

    parameters['input_path'] = input_paths['aeroplane']
    parameters['sampling_method'] = 'refined_convex_hull'
    graph_random = segment_into_planes(parameters, visualize=True)


