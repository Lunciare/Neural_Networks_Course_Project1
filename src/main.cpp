#include "Loader/MNISTLoader.h"
#include "LossFunctions/LossFunction.h"
#include "Model/Model.h"
#include "Optimizer/Optimizer.h"
#include "Tests/Tests.h"
#include "Utilities/FileWriter.h"
#include "Utilities/Random.h"
#include "Utilities/Utils.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>

using namespace neural_network;

int main() {
  test::runAllTests();

  std::vector<Vector> train_images, test_images;
  std::vector<int> train_labels, test_labels;
  if (!loadMNIST("../data/train-images.idx3-ubyte",
                 "../data/train-labels.idx1-ubyte", train_images,
                 train_labels) ||
      !loadMNIST("../data/t10k-images.idx3-ubyte",
                 "../data/t10k-labels.idx1-ubyte", test_images, test_labels)) {
    std::cerr << "Failed to load MNIST data!\n";
    return 1;
  }

  std::vector<Vector> train_targets, test_targets;
  for (int lbl : train_labels) {
    Vector y = Vector::Zero(10);
    y[lbl] = 1.0;
    train_targets.push_back(y);
  }
  for (int lbl : test_labels) {
    Vector y = Vector::Zero(10);
    y[lbl] = 1.0;
    test_targets.push_back(y);
  }

  std::cout << "Select model architecture:\n";
  std::cout << "1. One hidden layer (ReLU + Identity)\n";
  std::cout << "2. Two hidden layers (ReLU + Sigmoid + Identity)\n";
  std::cout << "3. Three hidden layers (ReLU + ReLU + ReLU + Softmax)\n";
  int choice;
  std::cout << "Enter choice (1/2/3): ";
  std::cin >> choice;

  int epochs;
  std::cout << "Enter number of training epochs: ";
  std::cin >> epochs;

  std::string model_name = "model" + std::to_string(choice);
  Model model =
      (choice == 1)
          ? Model({784, 128, 10}, {ActivationFunction::Type::ReLU,
                                   ActivationFunction::Type::Identity})
      : (choice == 2)
          ? Model({784, 64, 64, 10}, {ActivationFunction::Type::ReLU,
                                      ActivationFunction::Type::Sigmoid,
                                      ActivationFunction::Type::Identity})
          : Model({784, 128, 64, 32, 10}, {ActivationFunction::Type::ReLU,
                                           ActivationFunction::Type::ReLU,
                                           ActivationFunction::Type::ReLU,
                                           ActivationFunction::Type::Softmax});

  Optimizer opt = Optimizer::Adam(0.001, 0.9, 0.999, 1e-8);

  std::ofstream train_loss_file("loss_" + model_name + "_train.csv");
  std::ofstream val_loss_file("loss_" + model_name + "_val.csv");
  std::ofstream acc_file("accuracy_" + model_name + ".csv");

  train_loss_file << "Epoch,Loss\n";
  val_loss_file << "Epoch,Loss\n";
  acc_file << "Epoch,Accuracy\n";

  const int barWidth = 30;

  std::cout << "\n=== Training " << model_name << " for " << epochs
            << " epoch(s) ===\n";

  std::default_random_engine rng(std::random_device{}());

  for (int e = 0; e < epochs; ++e) {
    // Shuffle indices
    std::vector<int> indices(train_images.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);

    double running_loss = 0.0;
    for (int i = 0; i < int(train_images.size()); ++i) {
      int idx = indices[i];

      if (choice == 3) {
        model.trainStep(train_images[idx], train_targets[idx],
                        LossFunction::crossEntropyGrad, opt);
        Vector out = model.forward(train_images[idx]);
        running_loss += LossFunction::crossEntropy(out, train_targets[idx]);
      } else {
        model.trainStep(train_images[idx], train_targets[idx],
                        LossFunction::mseGrad, opt);
        Vector out = model.forward(train_images[idx]);
        running_loss += LossFunction::mse(out, train_targets[idx]);
      }

      float fraction = float(i + 1) / train_images.size();
      int pos = int(barWidth * fraction);
      int percent = int(fraction * 100.0f);

      std::cout << "\r[";
      for (int j = 0; j < barWidth; ++j)
        std::cout << (j < pos ? '=' : ' ');
      std::cout << "] " << std::setw(3) << percent << "% "
                << "(" << (i + 1) << "/" << train_images.size() << ") "
                << "L:" << std::fixed << std::setprecision(4)
                << (running_loss / (i + 1)) << std::flush;
    }

    double avg_train_loss = running_loss / train_images.size();
    train_loss_file << (e + 1) << "," << avg_train_loss << "\n";

    double val_loss = 0.0;
    int correct = 0;
    for (int i = 0; i < int(test_images.size()); ++i) {
      Vector out = model.forward(test_images[i]);
      if (choice == 3)
        val_loss += LossFunction::crossEntropy(out, test_targets[i]);
      else
        val_loss += LossFunction::mse(out, test_targets[i]);

      Eigen::Index predIndex;
      out.maxCoeff(&predIndex);
      if (predIndex == test_labels[i])
        ++correct;
    }

    double avg_val_loss = val_loss / test_images.size();
    double accuracy = 100.0 * double(correct) / test_images.size();

    val_loss_file << (e + 1) << "," << avg_val_loss << "\n";
    acc_file << (e + 1) << "," << accuracy << "\n";

    std::cout << "\nEpoch " << (e + 1)
              << " finished. Train Loss: " << avg_train_loss
              << ", Val Loss: " << avg_val_loss << ", Accuracy: " << accuracy
              << "%\n";
  }

  train_loss_file.close();
  val_loss_file.close();
  acc_file.close();

  FileWriter out("model_" + model_name + ".bin");
  out << model;

  return 0;
}
