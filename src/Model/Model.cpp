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

// Прямой проход (inference). Теперь НЕ const, чтобы мы могли вызывать
// forwardTrain() на слоях
Vector Model::forward(const Vector &input) {
  Vector x = input;
  for (auto &layer : layers_) {
    // forwardTrain() сохраняет внутренние кэши в слое
    x = layer.forwardTrain(x);
  }
  return x;
}

// Прямой проход при обучении (gather all activations). Не const, т.к.
// forwardTrain() меняет слой.
std::vector<Vector> Model::forwardTrain(const Vector &x) {
  std::vector<Vector> activations;
  activations.reserve(layers_.size() + 1);

  Vector current = x;
  activations.push_back(current);

  for (auto &layer : layers_) {
    current = layer.forwardTrain(current);
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
  // 1) Инициализируем Optimizer-кэш в каждом слое
  for (auto &layer : layers_) {
    layer.set_cache(optimizer);
  }

  // 2) Делаем forwardTrain, чтобы посчитать все активации и получить градиент
  // на выходе
  auto activations = forwardTrain(x);
  Vector grad = lossGrad(activations.back(), y);

  // 3) Backward: в каждом Layer::backward(...) вызывается optimizer.update(...)
  backward(grad, optimizer);

  // 4) Очистим кэш каждого слоя (иначе данные будут “висячими” до следующего
  // trainStep)
  for (auto &layer : layers_) {
    layer.free_cache();
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

// ==================== Реализация шаблонных методов ====================
// (ранее вызывала ошибку “no member named read/write”)
template <class Reader> void Model::read(Reader &in) { in >> layers_; }

template <class Writer> void Model::write(Writer &out) const { out << layers_; }

// Явные инстанцирования шаблонов (чтобы не было “undefined reference”)
template void Model::read(FileReader &);
template void Model::write(FileWriter &) const;

} // namespace neural_network
