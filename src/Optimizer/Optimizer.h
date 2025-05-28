#pragma once

#include "Utilities/Utils.h"
#include <functional>
#include <memory>

namespace neural_network {

class Optimizer {
public:
  static Optimizer SGD(double lr);
  static Optimizer Adam(double lr, double beta1 = 0.9, double beta2 = 0.999,
                        double eps = 1e-8);

  void update(Matrix &param, Matrix &m, Matrix &v, const Matrix &grad,
              Index iteration) const;
  void update(Vector &param, Vector &m, Vector &v, const Vector &grad,
              Index iteration) const;

private:
  std::function<void(Matrix &, Matrix &, Matrix &, const Matrix &, Index)>
      update_matrix_;
  std::function<void(Vector &, Vector &, Vector &, const Vector &, Index)>
      update_vector_;

  Optimizer(
      std::function<void(Matrix &, Matrix &, Matrix &, const Matrix &, Index)>
          um,
      std::function<void(Vector &, Vector &, Vector &, const Vector &, Index)>
          uv);
};

} // namespace neural_network
