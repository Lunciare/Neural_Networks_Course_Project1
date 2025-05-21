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

  // Training code ---
  std::vector<Eigen::VectorXd> train_images;
  std::vector<int> train_labels;

  if (!loadMNIST("../data/train-images.idx3-ubyte",
                 "../data/train-labels.idx1-ubyte", train_images,
                 train_labels)) {
    std::cerr << "Failed to load MNIST data!\n";
    return 1;
  }
  std::cout << "Train size: " << train_images.size() << std::endl;

  // -- Только для быстрой проверки: взять подмножество данных (например, 1000
  // картинок) --
  const size_t N = 1000; // поменять на 60000 для полного обучения
  std::vector<Eigen::VectorXd> images_short(train_images.begin(),
                                            train_images.begin() + N);
  std::vector<int> labels_short(train_labels.begin(), train_labels.begin() + N);

  // Adam optimizer (or use Optimizer opt(0.01); for SGD)
  Optimizer opt(0.001, 0.9, 0.999, 1e-8);

  Model model(
      {784, 128, 10},
      {ActivationFunction::Type::ReLU, ActivationFunction::Type::Identity},
      opt);

  for (int epoch = 0; epoch < 1; ++epoch) {
    double loss = 0.0;
    for (size_t i = 0; i < images_short.size(); ++i) {
      Eigen::VectorXd x = images_short[i];
      Eigen::VectorXd y = Eigen::VectorXd::Zero(10);
      y[labels_short[i]] = 1.0;
      model.trainStep(x, y);
      loss += LossFunction::mse(model.forward(x), y);

      // -- Прогресс-бар --
      if (i % 100 == 0 || i == images_short.size() - 1) {
        int barWidth = 40;
        float progress = float(i + 1) / images_short.size();
        std::cout << "\r[";
        int pos = barWidth * progress;
        for (int j = 0; j < barWidth; ++j) {
          if (j < pos)
            std::cout << "=";
          else if (j == pos)
            std::cout << ">";
          else
            std::cout << " ";
        }
        std::cout << "] " << int(progress * 100.0) << "% (" << (i + 1) << "/"
                  << images_short.size() << ")" << std::flush;
      }
    }
    std::cout << std::endl;
    std::cout << "Epoch " << (epoch + 1)
              << ", MSE: " << (loss / images_short.size()) << std::endl;
  }
}
