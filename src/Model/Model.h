#pragma once

#include "Layers/Layer.h"
#include "LossFunctions/LossFunction.h"
#include <initializer_list>
#include <vector>

namespace neural_network {

class Model {
public:
  Model(std::initializer_list<size_t> layer_sizes,
        std::initializer_list<ActivationFunction::Type> activations);

  Vector forward(const Vector &input) const;

  void train(const std::vector<Vector> &xs, const std::vector<Vector> &ys,
             int epochs, LossFunction loss, Optimizer &optimizer);

  const std::vector<Layer> &layers() const { return layers_; }

private:
  std::vector<Layer> layers_;

  std::vector<Vector> forwardTrain(const Vector &x) const;
  void backward(const std::vector<Vector> &activations, const Vector &grad);

  void trainStep(const Vector &x, const Vector &y, LossFunction loss,
                 Optimizer &optimizer);
};

} // namespace neural_network
