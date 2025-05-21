#include "Random.h"

namespace neural_network
{

Random global_random;

Random::Random(int seed) : generator_(seed) {}

Random::Matrix Random::uniformMatrix(size_t rows, size_t cols, double a,
                                     double b)
{
    std::uniform_real_distribution<double> dist(a, b);
    Matrix mat(rows, cols);
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            mat(i, j) = dist(generator_);
    return mat;
}

Random::Vector Random::uniformVector(size_t size, double a, double b)
{
    std::uniform_real_distribution<double> dist(a, b);
    Vector vec(size);
    for (size_t i = 0; i < size; ++i)
        vec(i) = dist(generator_);
    return vec;
}

}// namespace neural_network
