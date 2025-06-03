# Neural Networks from Scratch

This project implements a fully connected neural network from scratch in C++ using the MNIST dataset for training and evaluation. The implementation includes customizable activation functions, optimizers (SGD, Adam), and loss functions (MSE, Cross-Entropy).

> **Note**: All source files are located on the `dev` branch. Please switch to the `dev` branch to access the complete implementation.

## Features

- Written in C++ with Eigen and EigenRand libraries
- Custom implementation of:
  - Forward and backward propagation
  - Optimizers (SGD, Adam)
  - Activation functions (ReLU, Sigmoid, Tanh, Identity)
  - Loss functions (MSE, Cross-Entropy)
- Data loading for the MNIST dataset
- Test accuracy evaluation
- Logging training loss to `loss.csv`
- Confusion matrix generation

## Requirements

- CMake ≥ 3.10
- C++20-compatible compiler (e.g., `clang++`, `g++`)
- Eigen (included via submodule)
- MNIST dataset files placed in `data/`:
  - `train-images.idx3-ubyte`
  - `train-labels.idx1-ubyte`
  - `t10k-images.idx3-ubyte`
  - `t10k-labels.idx1-ubyte`

## Build Instructions

```bash
# Clone the repository and switch to dev branch
git clone https://github.com/Lunciare/Neural_Networks_Course_Project1.git
cd Neural_Networks_Course_Project1
git checkout dev

# Initialize submodules (e.g., Eigen)
git submodule update --init --recursive

# Create build directory and compile the project
mkdir build && cd build
cmake ..
make
```
## Run the Project

```bash
# After building, run the executable:
./neural_net
```
## Output

- Console output will show training progress and final test accuracy.
- A CSV file loss.csv will be generated, containing the average loss per training step.
- A predictions.csv file will be created for further evaluation like plotting a confusion matrix.

## Author

Alexandra Suvorova

## Supervisor

Dmitriy Trushin