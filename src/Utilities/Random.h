#pragma once
#include "eigen/Eigen/Dense"
#include <random>

namespace neural_network
{

class Random
{
public:
    using Matrix = Eigen::MatrixXd;
    using Vector = Eigen::VectorXd;

    explicit Random(int seed = kDefaultSeed);

    // Generates a matrix of shape [rows x cols] with uniform random values in [a,
    // b]
    Matrix uniformMatrix(size_t rows, size_t cols, double a, double b);

    // Generates a vector of length size with uniform random values in [a, b]
    Vector uniformVector(size_t size, double a, double b);

    static constexpr int kDefaultSeed = 42;

private:
    std::mt19937 generator_;
};

extern Random global_random;

}// namespace neural_network
