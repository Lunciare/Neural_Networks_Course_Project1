#include "Tests.h"
#include "ActivationFunction.h"
#include "Layer.h"
#include "Optimizer.h"
#include <cassert>
#include <iostream>

namespace neural_network
{
namespace test
{

bool testActivationFunction()
{
    using AF = ActivationFunction;
    Eigen::VectorXd x(3);
    x << -1, 0, 2;
    Eigen::VectorXd y_relu = AF::apply(AF::Type::ReLU, x);
    if (!(y_relu.array() == Eigen::ArrayXd({0, 0, 2})).all())
        return false;

    Eigen::VectorXd y_sigmoid = AF::apply(AF::Type::Sigmoid, x);
    if (std::abs(y_sigmoid[0] - 0.26894) > 1e-4)
        return false;

    Eigen::VectorXd y_tanh = AF::apply(AF::Type::Tanh, x);
    if (std::abs(y_tanh[2] - std::tanh(2)) > 1e-8)
        return false;

    return true;
}

bool testOptimizerSGD()
{
    Optimizer opt(0.1);// SGD
    Eigen::MatrixXd w = Eigen::MatrixXd::Ones(2, 2);
    Eigen::MatrixXd m = Eigen::MatrixXd::Zero(2, 2);
    Eigen::MatrixXd v = Eigen::MatrixXd::Zero(2, 2);
    Eigen::MatrixXd grad = Eigen::MatrixXd::Ones(2, 2);

    opt.update(w, m, v, grad, 1);
    if (std::abs(w(0, 0) - 0.9) > 1e-9)
        return false;
    return true;
}

bool testOptimizerAdam()
{
    Optimizer opt(0.1, 0.9, 0.999, 1e-8);// Adam
    Eigen::MatrixXd w = Eigen::MatrixXd::Ones(1, 1);
    Eigen::MatrixXd m = Eigen::MatrixXd::Zero(1, 1);
    Eigen::MatrixXd v = Eigen::MatrixXd::Zero(1, 1);
    Eigen::MatrixXd grad = Eigen::MatrixXd::Ones(1, 1);

    opt.update(w, m, v, grad, 1);
    // Adam first step: delta ≈ 0.1 / (1 / sqrt(1) + 1e-8) ≈ 0.1
    if (std::abs(w(0, 0) - (1.0 - 0.1)) > 1e-5)
        return false;
    return true;
}

bool testLayerForwardBackward()
{
    Optimizer opt(0.01, 0.9, 0.999, 1e-8);// Adam
    Layer l(2, 1, ActivationFunction::Type::Identity, opt);

    Eigen::VectorXd input(2);
    input << 1.0, -1.0;

    Eigen::VectorXd output = l.forward(input);
    Eigen::VectorXd grad_out(1);
    grad_out << 1.0;
    Eigen::VectorXd grad_in = l.backward(grad_out);
    // Just check code runs and output size is correct
    if (grad_in.size() != 2)
        return false;
    return true;
}

bool runAllTests()
{
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

}// namespace test
}// namespace neural_network
