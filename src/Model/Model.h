#pragma once
#include "Layers/Layer.h"
#include <vector>

namespace neural_network {

class Model {
public:
  Model(const std::vector<size_t> &layer_sizes,
        const std::vector<ActivationFunction::Type> &activations,
        const Optimizer &optimizer);

  Vector forward(const Vector &input);

  void trainStep(const Vector &x, const Vector &y);

  bool save(const std::string &prefix) const;
  bool load(const std::string &prefix);

  const std::vector<Layer> &layers() const { return layers_; }

private:
  std::vector<Layer> layers_;
};

} // namespace neural_network
