#pragma once
#include "Layers/Layer.h"
#include <vector>

namespace neural_network {

// Neural network as a sequence of layers with shared optimizer
class Model {
public:
  Model(const std::vector<size_t> &layer_sizes,
        const std::vector<ActivationFunction::Type> &activations,
        const Optimizer &optimizer);

  Vector forward(const Vector &input);

  // One training step: forward, backward and optimizer update
  void trainStep(const Vector &x, const Vector &y);

  bool save(const std::string &prefix) const;
  bool load(const std::string &prefix);

  const std::vector<Layer> &layers() const { return layers_; }

private:
  std::vector<Layer> layers_;
};

} // namespace neural_network
