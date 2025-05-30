#pragma once

#include "ActivationFunctions/ActivationFunction.h"
#include "Optimizer/Optimizer.h"
#include "Utilities/Utils.h"

#include <string>

namespace neural_network {

class Layer {
public:
  Layer(Index in, Index out, ActivationFunction activation);

  Vector forward(const Vector &input) const;
  Vector forwardTrain(const Vector &input); // saves input/z
  Vector backward(const Vector &grad_output);

  void setOptimizer(Optimizer *optimizer);

  enum class IOStatus { OK, IOError };
  IOStatus saveWeights(const std::string &filename) const;
  IOStatus loadWeights(const std::string &filename);

private:
  static Matrix initWeights(Index out, Index in);
  static Vector initBiases(Index out);

  ActivationFunction::Type activation_type_;
  ActivationFunction activation_;

  Matrix weights_;
  Vector biases_;

  // Adam optimizer moments
  Matrix m_w_;
  Matrix v_w_;
  Vector m_b_;
  Vector v_b_;
  Index t_;

  // Caches for backprop
  Vector last_input_;
  Vector last_z_;

  Optimizer *optimizer_;
};

} // namespace neural_network
