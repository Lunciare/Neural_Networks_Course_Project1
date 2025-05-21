#pragma once
#include <Eigen/Dense>
#include <string>
#include <vector>

namespace neural_network {

// Loads the MNIST dataset from binary files.
// Each image is a flattened Eigen::VectorXd (values normalized to [0,1]).
bool loadMNIST(const std::string &image_file, const std::string &label_file,
               std::vector<Eigen::VectorXd> &images, std::vector<int> &labels);

} // namespace neural_network
