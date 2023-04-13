from parameters import generate_parameters_to_experiment

from main import segment_into_planes

import matplotlib.pyplot as plt
import pprint

import numpy as np

parameters_to_experiment = generate_parameters_to_experiment()

pprint.pprint(parameters_to_experiment)


# subfigures = plt.subplots(1, 2, figsize=(10, 5))

# subfigures[1][0].plot(graph1['iterations'], graph1['inliers'])

for parameters in parameters_to_experiment:

    np.random.seed(0)

    parameters['sampling_method'] = 'random'
    graph_random = segment_into_planes(parameters)
    np.random.seed(0)

    parameters['sampling_method'] = 'convex_hull'
    graph_convex_hull = segment_into_planes(parameters)

    np.random.seed(0)

    parameters['sampling_method'] = 'refined_convex_hull'
    graph_refined_convex_hull = segment_into_planes(parameters)


    fig, ax = plt.subplots()
    a = ax.plot(graph_random['iterations'], graph_random['inliers'])
    b = ax.plot(graph_convex_hull['iterations'], graph_convex_hull['inliers'])
    c = ax.plot(graph_refined_convex_hull['iterations'], graph_refined_convex_hull['inliers'])

    # label lines
    ax.legend((a[0], b[0], c[0]), ('random', 'convex_hull', 'refined_convex_hull'), loc='best')

    # label axes
    ax.set_xlabel('iterations')
    ax.set_ylabel('inliers')
    fig.savefig(f"noisy_results/{parameters['object_name']}.png")
    plt.close(fig)
    del fig






