#include "Tests/Tests.h"
#include "ActivationFunctions/ActivationFunction.h"
#include "Layers/Layer.h"
#include "Optimizer/Optimizer.h"
#include <cassert>
#include <iostream>

namespace {

using namespace neural_network;
using namespace neural_network::test;

TestStatus testActivationFunction() {
  using AF = ActivationFunction;
  Vector x(3);
  x << -1, 0, 2;

  Vector y_relu = AF::create(AF::Type::ReLU).apply(x);
  Eigen::ArrayXd expected(3);
  expected << 0, 0, 2;
  if (!(y_relu.array() == expected).all()) {
    std::cout << "[FAIL] ActivationFunction::ReLU output mismatch\n";
    return TestStatus::Error;
  }

  Vector y_sigmoid = AF::create(AF::Type::Sigmoid).apply(x);
  if (std::abs(y_sigmoid[0] - 0.26894) > 1e-4) {
    std::cout << "[FAIL] ActivationFunction::Sigmoid value incorrect\n";
    return TestStatus::Error;
  }

  Vector y_tanh = AF::create(AF::Type::Tanh).apply(x);
  if (std::abs(y_tanh[2] - std::tanh(2)) > 1e-8) {
    std::cout << "[FAIL] ActivationFunction::Tanh value incorrect\n";
    return TestStatus::Error;
  }

  return TestStatus::OK;
}

TestStatus testOptimizerSGD() {
  Optimizer opt = Optimizer::SGD(0.1);
  Matrix w = Matrix::Ones(2, 2);
  Matrix grad = Matrix::Ones(2, 2);
  std::any cache = opt.init_cache(2, 2);

  opt.update(w, cache, grad);

  return !(std::abs(w(0, 0) - 0.9) > 1e-9)
             ? TestStatus::OK
             : (std::cout << "[FAIL] Optimizer::SGD weight update incorrect\n",
                TestStatus::Error);
}

TestStatus testOptimizerAdam() {
  Optimizer opt = Optimizer::Adam(0.1, 0.9, 0.999, 1e-8);
  Matrix w = Matrix::Ones(1, 1);
  Matrix grad = Matrix::Ones(1, 1);

  std::any cache = opt.init_cache(w.rows(), w.cols());
  opt.update(w, cache, grad);

  if (w(0, 0) >= 1.0 || w(0, 0) <= 0.98) {
    std::cout << "[FAIL] Optimizer::Adam weight update incorrect, got w = "
              << w(0, 0) << "\n";
    return TestStatus::Error;
  }

  return TestStatus::OK;
}

TestStatus testLayerForwardBackward() {
  Optimizer opt = Optimizer::Adam(0.01, 0.9, 0.999, 1e-8);
  Layer l(In(2), Out(1),
          ActivationFunction::create(ActivationFunction::Type::Identity));
  l.set_cache(opt);

  Vector input(2);
  input << 1.0, -1.0;

  Vector output = l.forward(input);
  Vector grad_out(1);
  grad_out << 1.0;
  Vector grad_in = l.backward(grad_out, opt);

  if (grad_in.size() != 2) {
    std::cout << "[FAIL] Layer::backward output size mismatch\n";
    return TestStatus::Error;
  }
  return TestStatus::OK;
}

} // anonymous namespace

namespace neural_network {
namespace test {

void runAllTests() {
  if (testActivationFunction() == TestStatus::Error)
    return;
  if (testOptimizerSGD() == TestStatus::Error)
    return;
  if (testOptimizerAdam() == TestStatus::Error)
    return;
  if (testLayerForwardBackward() == TestStatus::Error)
    return;

  std::cout << "[OK] All tests passed!\n";
}

} // namespace test
} // namespace neural_network
