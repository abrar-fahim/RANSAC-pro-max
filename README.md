# RANSAC-pro-max
Improved RANSAC for fitting planes to point clouds




Dataset: https://www.kaggle.com/code/kerneler/starter-shapenet-0b740c60-8/input


## Instructions on running our algorithm

- The [main.py](main.py) file contains our `segment_into_planes` function, which takes `parameters` dictionary as input

- Running the [main.py](main.py) file runs our code
- The [parameters.py](parameters.py) contains the `parameters` dictionary, which contains the algorithm parameters that `segment_into_planes` function uses to run our modified RANSAC algorithm.

### Inside the `parameters` dictionary:

- change the `input_path` value to change the object that the algorithm segments into planes. We included the objects mug, aeroplane, table, skateboard, and laptop in this repository to make it easy to run without having to download the entire dataset.

- For instance, to run the algorithm with the mug object, set `input_path` in `parameters` dictionary to `input_paths['mug']`


