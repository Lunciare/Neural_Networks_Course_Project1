#include "Loader/MNISTLoader.h"
#include "LossFunctions/LossFunction.h"
#include "Model/Model.h"
#include "Optimizer/Optimizer.h"
#include "Tests/Tests.h"
#include "Utilities/Random.h"
#include "Utilities/Utils.h"

#include <iomanip>
#include <iostream>

using namespace neural_network;

int main() {
  test::runAllTests();

  std::vector<Vector> train_images;
  std::vector<int> train_labels;
  if (!loadMNIST("../data/train-images.idx3-ubyte",
                 "../data/train-labels.idx1-ubyte", train_images,
                 train_labels)) {
    std::cerr << "Failed to load MNIST training data!\n";
    return 1;
  }
  const int N = static_cast<int>(train_images.size());
  std::cout << "Train size: " << N << "\n";

  Optimizer opt = Optimizer::Adam(0.001, 0.9, 0.999, 1e-8);
  Model model({784, 128, 10}, {ActivationFunction::Type::ReLU,
                               ActivationFunction::Type::Identity});

  std::vector<Vector> train_targets;
  train_targets.reserve(train_labels.size());
  for (int lbl : train_labels) {
    Vector y = Vector::Zero(10);
    y[lbl] = 1.0;
    train_targets.push_back(y);
  }

  const int barWidth = 30;
  double running_loss = 0.0;

  std::cout << "\n=== Training (1 epoch) ===\n";

  for (int i = 0; i < N; ++i) {
    model.trainStep(train_images[i], train_targets[i], LossFunction::mseGrad,
                    opt);

    Vector out = model.forward(train_images[i]);
    double loss_i = LossFunction::mse(out, train_targets[i]);
    running_loss += loss_i;
    double avgLoss = running_loss / (i + 1);

    float fraction = static_cast<float>(i + 1) / N;
    int pos = static_cast<int>(barWidth * fraction);
    int percent = static_cast<int>(fraction * 100.0f);

    std::cout << "\r[";
    for (int j = 0; j < barWidth; ++j) {
      std::cout << (j < pos ? '=' : ' ');
    }

    std::cout << "] " << std::setw(3) << percent << "% "
              << "(" << (i + 1) << "/" << N << ") "
              << "L:" << std::fixed << std::setprecision(4) << avgLoss
              << std::flush;
  }

  std::cout << "\n\n=== Training finished ===\n";

  std::vector<Vector> test_images;
  std::vector<int> test_labels;
  if (!loadMNIST("../data/t10k-images.idx3-ubyte",
                 "../data/t10k-labels.idx1-ubyte", test_images, test_labels)) {
    std::cerr << "Failed to load MNIST test data!\n";
    return 1;
  }
  std::cout << "Test size: " << test_images.size() << "\n";

  int correct = 0;
  for (size_t i = 0; i < test_images.size(); ++i) {
    Vector out = model.forward(test_images[i]);
    Eigen::Index predIndex;
    out.maxCoeff(&predIndex);
    if (static_cast<int>(predIndex) == test_labels[i]) {
      ++correct;
    }
  }
  double accuracy = 100.0 * static_cast<double>(correct) / test_images.size();
  std::cout << "\nTest Accuracy: " << std::fixed << std::setprecision(4)
            << accuracy << "%\n";

  return 0;
}
