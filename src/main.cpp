#include "Loader/MNISTLoader.h"
#include "LossFunctions/LossFunction.h"
#include "Model/Model.h"
#include "Optimizer/Optimizer.h"
#include "Tests/Tests.h"
#include "Utilities/Random.h"
#include "Utilities/Utils.h"
#include <iostream>

using namespace neural_network;

int main() {
  test::runAllTests();

  std::vector<Vector> train_images;
  std::vector<int> train_labels;

  if (!loadMNIST("../data/train-images.idx3-ubyte",
                 "../data/train-labels.idx1-ubyte", train_images,
                 train_labels)) {
    std::cerr << "Failed to load MNIST data!\n";
    return 1;
  }

  std::cout << "Train size: " << size(train_images) << std::endl;

  Optimizer opt = Optimizer::Adam(0.001, 0.9, 0.999, 1e-8);

  Model model(
      {784, 128, 10},
      {ActivationFunction::Type::ReLU, ActivationFunction::Type::Identity},
      opt);

  constexpr Index TRAIN_LIMIT = 2000;
  if (size(train_images) > TRAIN_LIMIT) {
    train_images.resize(TRAIN_LIMIT);
    train_labels.resize(TRAIN_LIMIT);
  }

  for (int epoch = 0; epoch < 1; ++epoch) {
    double loss = 0.0;

    for (Index i = 0; i < size(train_images); ++i) {
      Vector x = train_images[i];
      Vector y = Vector::Zero(10);
      y[train_labels[i]] = 1.0;

      model.trainStep(x, y);
      loss += LossFunction::mse(model.forward(x), y);

      if (i % 500 == 0 || i + 1 == size(train_images)) {
        int barWidth = 40;
        float progress = float(i + 1) / size(train_images);
        std::cout << "\r[";
        int pos = barWidth * progress;
        for (int j = 0; j < barWidth; ++j)
          std::cout << (j < pos ? "=" : " ");
        std::cout << "] " << int(progress * 100.0) << "% (" << (i + 1) << "/"
                  << size(train_images) << ")" << std::flush;
      }
    }

    std::cout << std::endl;
    std::cout << "Epoch " << (epoch + 1)
              << ", MSE: " << (loss / size(train_images)) << std::endl;
  }
}
