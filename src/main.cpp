#include "Loader/MNISTLoader.h"
#include "LossFunctions/LossFunction.h"
#include "Model/Model.h"
#include "Optimizer/Optimizer.h"
#include "Tests/Tests.h"
#include <iostream>

using namespace neural_network;

int main() {
  // Run tests before starting training
  if (!test::runAllTests()) {
    std::cerr << "Unit tests failed! Exiting.\n";
    return 1;
  }

  // --- Usual training code ---
  std::vector<Eigen::VectorXd> train_images;
  std::vector<int> train_labels;

  if (!loadMNIST("train-images-idx3-ubyte", "train-labels-idx1-ubyte",
                 train_images, train_labels)) {
    std::cerr << "Failed to load MNIST data!\n";
    return 1;
  }
  std::cout << "Train size: " << train_images.size() << std::endl;

  // Adam optimizer (or use Optimizer opt(0.01); for SGD)
  Optimizer opt(0.001, 0.9, 0.999, 1e-8);

  Model model(
      {784, 128, 10},
      {ActivationFunction::Type::ReLU, ActivationFunction::Type::Identity},
      opt);

  for (int epoch = 0; epoch < 1; ++epoch) {
    double loss = 0.0;
    for (size_t i = 0; i < train_images.size(); ++i) {
      Eigen::VectorXd x = train_images[i];
      Eigen::VectorXd y = Eigen::VectorXd::Zero(10);
      y[train_labels[i]] = 1.0;
      model.trainStep(x, y);
      loss += LossFunction::mse(model.forward(x), y);
    }
    std::cout << "Epoch " << (epoch + 1)
              << ", MSE: " << (loss / train_images.size()) << std::endl;
  }
}
