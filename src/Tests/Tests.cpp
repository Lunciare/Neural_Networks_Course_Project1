#include "Tests/Tests.h"
#include "ActivationFunctions/ActivationFunction.h"
#include "Layers/Layer.h"
#include "Optimizer/Optimizer.h"
#include <cassert>
#include <iostream>

namespace neural_network {
namespace test {

bool testActivationFunction() {
  using AF = ActivationFunction;
  Vector x(3);
  x << -1, 0, 2;
  Vector y_relu = AF::apply(AF::Type::ReLU, x);
  Eigen::ArrayXd expected(3);
  expected << 0, 0, 2;
  if (!(y_relu.array() == expected).all())
    return false;

  Vector y_sigmoid = AF::apply(AF::Type::Sigmoid, x);
  if (std::abs(y_sigmoid[0] - 0.26894) > 1e-4)
    return false;

  Vector y_tanh = AF::apply(AF::Type::Tanh, x);
  if (std::abs(y_tanh[2] - std::tanh(2)) > 1e-8)
    return false;

  return true;
}

bool testOptimizerSGD() {
  Optimizer opt(0.1); // SGD
  Matrix w = Matrix::Ones(2, 2);
  Matrix m = Matrix::Zero(2, 2);
  Matrix v = Matrix::Zero(2, 2);
  Matrix grad = Matrix::Ones(2, 2);

  opt.update(w, m, v, grad, 1);
  if (std::abs(w(0, 0) - 0.9) > 1e-9)
    return false;
  return true;
}

bool testOptimizerAdam() {
  Optimizer opt(0.1, 0.9, 0.999, 1e-8); // Adam
  Matrix w = Matrix::Ones(1, 1);
  Matrix m = Matrix::Zero(1, 1);
  Matrix v = Matrix::Zero(1, 1);
  Matrix grad = Matrix::Ones(1, 1);

  opt.update(w, m, v, grad, 1);
  // Adam first step: delta ≈ 0.1 / (1 / sqrt(1) + 1e-8) ≈ 0.1
  if (std::abs(w(0, 0) - (1.0 - 0.1)) > 1e-5)
    return false;
  return true;
}

bool testLayerForwardBackward() {
  Optimizer opt(0.01, 0.9, 0.999, 1e-8); // Adam
  Layer l(In(2), Out(1), ActivationFunction::Type::Identity, opt);

  Vector input(2);
  input << 1.0, -1.0;

  Vector output = l.forward(input);
  Vector grad_out(1);
  grad_out << 1.0;
  Vector grad_in = l.backward(grad_out);
  // Just check code runs and output size is correct
  if (grad_in.size() != 2)
    return false;
  return true;
}

bool runAllTests() {
  bool ok = true;
  ok = ok && testActivationFunction();
  ok = ok && testOptimizerSGD();
  ok = ok && testOptimizerAdam();
  ok = ok && testLayerForwardBackward();
  if (ok)
    std::cout << "[OK] All tests passed!\n";
  else
    std::cout << "[FAIL] Some tests failed!\n";
  return ok;
}

} // namespace test
} // namespace neural_network
