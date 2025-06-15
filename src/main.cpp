#include "Loader/MNISTLoader.h"
#include "LossFunctions/LossFunction.h"
#include "Model/Model.h"
#include "Optimizer/Optimizer.h"
#include "Tests/Tests.h"
#include "Utilities/FileReader.h"
#include "Utilities/FileWriter.h"
#include "Utilities/Random.h"
#include "Utilities/Utils.h"

#include <fstream>
#include <iomanip>
#include <iostream>

using namespace neural_network;

void trainAndEvaluate(
    Model &model, const std::vector<Vector> &train_images,
    const std::vector<Vector> &train_targets,
    const std::vector<Vector> &test_images, const std::vector<int> &test_labels,
    const std::string &model_name, int epochs,
    const std::function<Vector(const Vector &, const Vector &)> &lossGrad,
    const std::function<double(const Vector &, const Vector &)> &lossFunc) {
  Optimizer opt = Optimizer::Adam(0.001, 0.9, 0.999, 1e-8);

  const int N = train_images.size();
  const int barWidth = 30;
  double running_loss = 0.0;
  std::ofstream loss_file("loss_" + model_name + ".csv");
  loss_file << "Step,Loss\n";

  std::cout << "\n=== Training " << model_name << " for " << epochs
            << " epoch(s) ===\n";

  for (int e = 0; e < epochs; ++e) {
    running_loss = 0.0;
    for (int i = 0; i < N; ++i) {
      model.trainStep(train_images[i], train_targets[i], lossGrad, opt);

      Vector out = model.forward(train_images[i]);
      double loss_i = lossFunc(out, train_targets[i]);
      running_loss += loss_i;
      double avgLoss = running_loss / (i + 1);
      loss_file << (e * N + i + 1) << "," << avgLoss << "\n";

      float fraction = float(i + 1) / N;
      int pos = int(barWidth * fraction);
      int percent = int(fraction * 100.0f);
      std::cout << "\r[";
      for (int j = 0; j < barWidth; ++j)
        std::cout << (j < pos ? '=' : ' ');
      std::cout << "] " << std::setw(3) << percent << "% "
                << "(" << (i + 1) << "/" << N << ") "
                << "L:" << std::fixed << std::setprecision(4) << avgLoss
                << std::flush;
    }
    std::cout << "\nEpoch " << (e + 1) << " finished.\n";
  }

  loss_file.close();
  {
    FileWriter fw("model_" + model_name + ".bin");
    fw << model;
  }

  // Accuracy evaluation
  int correct = 0;
  std::ofstream cm_file("predictions_" + model_name + ".csv");
  cm_file << "True,Predicted\n";

  for (size_t i = 0; i < test_images.size(); ++i) {
    Vector out = model.forward(test_images[i]);
    Eigen::Index predIndex;
    out.maxCoeff(&predIndex);
    int predicted = predIndex;
    int truth = test_labels[i];

    cm_file << truth << "," << predicted << "\n";
    if (predicted == truth)
      ++correct;
  }
  cm_file.close();

  double accuracy = 100.0 * double(correct) / test_images.size();
  std::cout << "\n"
            << model_name << " Test Accuracy: " << std::fixed
            << std::setprecision(4) << accuracy << "%\n";
}

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

  std::vector<Vector> train_targets;
  train_targets.reserve(train_labels.size());
  for (int lbl : train_labels) {
    Vector y = Vector::Zero(10);
    y[lbl] = 1.0;
    train_targets.push_back(y);
  }

  std::vector<Vector> test_images;
  std::vector<int> test_labels;
  if (!loadMNIST("../data/t10k-images.idx3-ubyte",
                 "../data/t10k-labels.idx1-ubyte", test_images, test_labels)) {
    std::cerr << "Failed to load MNIST test data!\n";
    return 1;
  }

  int epochs = 1;
  std::cout << "Enter number of epochs: ";
  std::cin >> epochs;

  Model model1({784, 128, 10}, {ActivationFunction::Type::ReLU,
                                ActivationFunction::Type::Identity});
  Model model2({784, 128, 64, 10},
               {ActivationFunction::Type::ReLU, ActivationFunction::Type::Tanh,
                ActivationFunction::Type::Identity});
  Model model3({784, 256, 128, 64, 10},
               {ActivationFunction::Type::ReLU, ActivationFunction::Type::ReLU,
                ActivationFunction::Type::Sigmoid,
                ActivationFunction::Type::Identity});

  trainAndEvaluate(model1, train_images, train_targets, test_images,
                   test_labels, "model1", epochs, LossFunction::mseGrad,
                   LossFunction::mse);
  trainAndEvaluate(model2, train_images, train_targets, test_images,
                   test_labels, "model2", epochs, LossFunction::mseGrad,
                   LossFunction::mse);
  trainAndEvaluate(model3, train_images, train_targets, test_images,
                   test_labels, "model3", epochs,
                   LossFunction::crossEntropyGrad, LossFunction::crossEntropy);

  return 0;
}
