#pragma once
#include "Utilities/Utils.h"
#include <string>
#include <vector>

namespace neural_network {

bool loadMNIST(const std::string &image_file, const std::string &label_file,
               std::vector<Vector> &images, std::vector<int> &labels);

} // namespace neural_network
