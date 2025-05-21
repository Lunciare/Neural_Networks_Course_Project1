#include "Model.h"
#include "LossFunction.h"
#include <cassert>

namespace neural_network
{

Model::Model(const std::vector<size_t>& layer_sizes,
             const std::vector<ActivationFunction::Type>& activations,
             const Optimizer& optimizer)
{
    assert(layer_sizes.size() == activations.size() + 1);
    for (size_t i = 1; i < layer_sizes.size(); ++i)
    {
        layers_.emplace_back(layer_sizes[i - 1], layer_sizes[i], activations[i - 1],
                             optimizer);
    }
}

Eigen::VectorXd Model::forward(const Eigen::VectorXd& input)
{
    Eigen::VectorXd x = input;
    for (auto& layer: layers_)
    {
        x = layer.forward(x);
    }
    return x;
}

void Model::trainStep(const Eigen::VectorXd& x, const Eigen::VectorXd& y)
{
    std::vector<Eigen::VectorXd> activations{x};
    for (auto& layer: layers_)
        activations.push_back(layer.forward(activations.back()));

    Eigen::VectorXd grad = LossFunction::mseGrad(activations.back(), y);

    // Backward pass
    for (int i = int(layers_.size()) - 1; i >= 0; --i)
    {
        grad = layers_[i].backward(grad);
    }
}

bool Model::save(const std::string& prefix) const
{
    for (size_t i = 0; i < layers_.size(); ++i)
    {
        if (!layers_[i].saveWeights(prefix + "_layer" + std::to_string(i) + ".txt"))
            return false;
    }
    return true;
}

bool Model::load(const std::string& prefix)
{
    for (size_t i = 0; i < layers_.size(); ++i)
    {
        if (!layers_[i].loadWeights(prefix + "_layer" + std::to_string(i) + ".txt"))
            return false;
    }
    return true;
}

}// namespace neural_network
