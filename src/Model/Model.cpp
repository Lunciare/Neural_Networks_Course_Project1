#include "Model.h"
#include <algorithm>
#include <iostream>

namespace neural_network {

Model::Model() = default;

Model::Model(const std::string &model_name) : name(model_name) {}

void Model::add(AnyLayer *layer) { layers.push_back(layer); }

void Model::compile(LossFunction *lossFunction, Optimizer *opt) {
  loss = lossFunction;
  optimizer = opt;
}

void Model::train(const std::vector<std::vector<double>> &X,
                  const std::vector<std::vector<double>> &y, int epochs,
                  int batch_size) {
  for (int epoch = 0; epoch < epochs; ++epoch) {
    for (size_t i = 0; i < X.size(); i += batch_size) {
      std::vector<std::vector<double>> x_batch(
          X.begin() + i, X.begin() + std::min(X.size(), i + batch_size));
      std::vector<std::vector<double>> y_batch(
          y.begin() + i, y.begin() + std::min(y.size(), i + batch_size));

      std::vector<std::vector<double>> output = x_batch;
      for (auto *layer : layers) {
        output = layer->forward(output);
      }

      std::vector<std::vector<double>> grad = loss->derivative(output, y_batch);
      for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
        grad = (*it)->backward(grad);
      }

      for (auto *layer : layers) {
        layer->updateWeights(*optimizer);
      }
    }
    std::cout << "Epoch " << epoch + 1 << " completed." << std::endl;
  }
}

std::vector<std::vector<double>>
Model::predict(const std::vector<std::vector<double>> &X) {
  std::vector<std::vector<double>> output = X;
  for (auto *layer : layers) {
    output = layer->forward(output);
  }
  return output;
}

void Model::summary() const {
  std::cout << "Model summary: " << name << std::endl;
  std::cout << "Number of layers: " << layers.size() << std::endl;
  for (size_t i = 0; i < layers.size(); ++i) {
    std::cout << "  Layer " << i + 1 << ": " << typeid(*layers[i]).name()
              << std::endl;
  }
}

} // namespace neural_network
