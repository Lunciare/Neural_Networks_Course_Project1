#include "Model/Model.h"
#include "Utilities/FileReader.h"
#include "Utilities/FileWriter.h"
#include <cassert>

namespace neural_network {

Model::Model(std::initializer_list<size_t> layer_sizes,
             std::initializer_list<ActivationFunction::Type> activations) {
  assert(layer_sizes.size() == activations.size() + 1);

  auto it = activations.begin();
  for (auto i = layer_sizes.begin(); i + 1 != layer_sizes.end(); ++i, ++it) {
    ActivationFunction func = ActivationFunction::create(*it);
    layers_.emplace_back(In(*i), Out(*(i + 1)), func);
  }
}

Vector Model::forward(const Vector &input) {
  Vector x = input;
  for (auto &layer : layers_) {
    x = layer.forward(x);
  }
  return x;
}

std::vector<Vector> Model::forwardTrain(const Vector &x) {
  std::vector<Vector> activations;
  activations.reserve(layers_.size() + 1);

  Vector current = x;
  activations.push_back(current);

  for (auto &layer : layers_) {
    current = layer.forward(current);
    activations.push_back(current);
  }
  return activations;
}

void Model::backward(const Vector &grad, const Optimizer &opt) {
  Vector g = grad;
  for (int i = static_cast<int>(layers_.size()) - 1; i >= 0; --i) {
    g = layers_[i].backward(g, opt);
  }
}

void Model::trainStep(
    const Vector &x, const Vector &y,
    const std::function<Vector(const Vector &, const Vector &)> &lossGrad,
    Optimizer &optimizer) {
  for (auto &layer : layers_) {
    layer.setCache(optimizer);
  }

  auto activations = forwardTrain(x);
  Vector grad = lossGrad(activations.back(), y);

  backward(grad, optimizer);

  for (auto &layer : layers_) {
    layer.freeCache();
  }
}

void Model::train(const std::vector<Vector> &xs, const std::vector<Vector> &ys,
                  int epochs, LossFunction loss, Optimizer &optimizer) {
  assert(xs.size() == ys.size());
  for (int e = 0; e < epochs; ++e) {
    for (size_t i = 0; i < xs.size(); ++i) {
      trainStep(xs[i], ys[i], LossFunction::mseGrad, optimizer);
    }
  }
}

const std::vector<Layer> &Model::layers() const { return layers_; }

template <class Reader> void Model::read(Reader &in) { in >> layers_; }

template <class Writer> void Model::write(Writer &out) const { out << layers_; }

template void Model::read(FileReader &);
template void Model::write(FileWriter &) const;

} // namespace neural_network
