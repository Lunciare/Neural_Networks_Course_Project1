#pragma once

#include "Utilities/Utils.h"
#include <any>
#include <functional>

namespace neural_network {

class Optimizer {
public:
  using MatrixUpdate =
      std::function<void(Matrix &, std::any &, const Matrix &)>;
  using VectorUpdate =
      std::function<void(Vector &, std::any &, const Vector &)>;
  using CacheInit = std::function<std::any(int rows, int cols)>;

  static Optimizer SGD(double lr);
  static Optimizer Adam(double lr, double beta1 = 0.9, double beta2 = 0.999,
                        double eps = 1e-8);

  void update(Matrix &param, std::any &cache, const Matrix &grad) const;
  void update(Vector &param, std::any &cache, const Vector &grad) const;
  std::any init_cache(int rows, int cols) const;

private:
  MatrixUpdate update_matrix_;
  VectorUpdate update_vector_;
  CacheInit init_cache_;

  Optimizer(MatrixUpdate mu, VectorUpdate vu, CacheInit ci);
};

} // namespace neural_network
