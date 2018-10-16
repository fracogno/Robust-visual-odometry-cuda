# Robust Odometry Estimation for RGB-D Cameras (CUDA implementation)

Implementing the paper [Robust Odometry Estimation for RGB-D Cameras](https://vision.in.tum.de/_media/spezial/bib/kerl13icra.pdf) with CUDA support.

The original code for the CPU version is from Robert Maier (TUM - Chair for Computer Vision & Artificial Intelligence).

## Run the Odometry Estimation
You can either run it on a dataset, e.g. one of the [RGB-D SLAM Dataset and Benchmark](https://vision.in.tum.de/data/datasets/rgbd-dataset) datasets, or run it live on a kinect similiar camera.

The [main.cu](./src/main.cu) file reads images from a dataset and perform the Odometry Estimation on it. For doing it live on a kinect camera use the code in [main_kinect.cu](./src/main_kinect.cu).