#pragma once

#include "Layers/Layer.h"
#include "LossFunctions/LossFunction.h"

#include <functional>
#include <initializer_list>
#include <vector>

namespace neural_network {

class FileReader;
class FileWriter;

class Model {
public:
  Model(std::initializer_list<size_t> layer_sizes,
        std::initializer_list<ActivationFunction::Type> activations);

  Vector forward(const Vector &input);

  void trainStep(
      const Vector &x, const Vector &y,
      const std::function<Vector(const Vector &, const Vector &)> &lossGrad,
      Optimizer &optimizer);

  void train(const std::vector<Vector> &xs, const std::vector<Vector> &ys,
             int epochs, LossFunction loss, Optimizer &optimizer);

  const std::vector<Layer, Eigen::aligned_allocator<Layer>> &layers() const;

private:
  std::vector<Layer, Eigen::aligned_allocator<Layer>> layers_;

  std::vector<Vector> forwardTrain(const Vector &x);

  void backward(const Vector &grad, const Optimizer &opt);

  friend FileReader &operator>>(FileReader &, Model &);
  friend FileWriter &operator<<(FileWriter &, const Model &);
};

} // namespace neural_network
