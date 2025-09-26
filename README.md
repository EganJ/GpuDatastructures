# GpuDatastructures

## Installation Instructions

Requires a machine with an NVIDIA gpu. The following details how to set up the environment and project using conda, but conda is not strictly required.

Environment setup:
```
// Create environment as normal
conda create MyEnv && conda activate MyEnv
conda install -c nvidia cuda-toolkit
```

Building and running:
```
mkdir build
cmake . -B build
cd build && make
./main
```