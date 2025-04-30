#pragma once

#include "DimensionStruct.h"
#include "Math.h"
#include "Random.h"

namespace neural_network {
class DropoutLayer {
public:
  struct DropoutCache {
    Matrix mask;
  };

  DropoutLayer(In in_size, Out out_size, double rate);

  Matrix evaluate(const Matrix &input) const;
  Matrix evaluate(const Matrix &input, DropoutCache &cache) const;

  Matrix getBackpropError(const Matrix &a, const Matrix &z, const Matrix &b,
                          const DropoutCache &cache) const;
  void updateW(const Matrix &grad_diff, Matrix &memory, int time_step);
  void updateB(const Matrix &grad_diff, Matrix &memory, int time_step);

  Index getInputSize() const;
  Index getOutputSize() const;

private:
  double rate_;
  Index input_size_;
  Index output_size_;
};

} // namespace neural_network
