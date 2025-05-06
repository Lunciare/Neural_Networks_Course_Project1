#include <Eigen/Dense>
#include <EigenRand/EigenRand>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>

// Only run these tests if NEON vectorization is available
#ifdef EIGEN_VECTORIZE_NEON

/**
 * @brief Test suite for verifying Eigen's vectorized math operations
 *
 * These tests validate that Eigen's SIMD-accelerated math functions produce
 * results consistent with standard library implementations.
 */
class VectorizedMathTest : public ::testing::Test {
protected:
  // Common test data shared across multiple tests
  static constexpr int VECTOR_SIZE = 8;
  Eigen::ArrayXf test_inputs;

  void SetUp() override {
    // Initialize with values that test edge cases and normal operation
    test_inputs.resize(VECTOR_SIZE);
    test_inputs << -1.0f, // Negative input (should produce NaN for log/sqrt)
        0.0f,             // Zero
        0.1f,             // Small positive
        0.2f, 0.5f,
        1.0f, // Boundary case
        2.0f,
        5.0f; // Typical positive value
  }
};

/**
 * @brief Test vectorized natural logarithm operation
 */
TEST_F(VectorizedMathTest, LogarithmOperation) {
  Eigen::ArrayXf eigen_results(VECTOR_SIZE);
  const auto &x = test_inputs;

  // Perform vectorized log operations
  eigen_results.writePacket<0>(0, Eigen::internal::plog(x.packet<0>(0)));
  eigen_results.writePacket<0>(4, Eigen::internal::plog(x.packet<0>(4)));

  // Debug output
  std::cout << "Log results: " << eigen_results.transpose() << std::endl;

  // Validate each result
  for (int i = 0; i < VECTOR_SIZE; ++i) {
    const float expected = std::log(x[i]);
    if (std::isnan(expected)) {
      EXPECT_TRUE(std::isnan(eigen_results[i]));
    } else {
      EXPECT_FLOAT_EQ(expected, eigen_results[i]);
    }
  }
}

/**
 * @brief Test vectorized sine operation
 */
TEST_F(VectorizedMathTest, SineOperation) {
  Eigen::ArrayXf eigen_results(VECTOR_SIZE);
  const auto &x = test_inputs;

  // Perform vectorized sin operations
  eigen_results.writePacket<0>(0, Eigen::internal::psin(x.packet<0>(0)));
  eigen_results.writePacket<0>(4, Eigen::internal::psin(x.packet<0>(4)));

  std::cout << "Sine results: " << eigen_results.transpose() << std::endl;

  for (int i = 0; i < VECTOR_SIZE; ++i) {
    const float expected = std::sin(x[i]);
    if (std::isnan(expected)) {
      EXPECT_TRUE(std::isnan(eigen_results[i]));
    } else {
      EXPECT_FLOAT_EQ(expected, eigen_results[i]);
    }
  }
}

/**
 * @brief Test combined vectorized sine and cosine operations
 */
TEST_F(VectorizedMathTest, CombinedSinCosOperation) {
  Eigen::ArrayXf sin_results(VECTOR_SIZE), cos_results(VECTOR_SIZE);
  const auto &x = test_inputs;
  Eigen::internal::Packet4f s, c;

  // Perform vectorized sincos operations
  Eigen::internal::psincos(x.packet<0>(0), s, c);
  sin_results.writePacket<0>(0, s);
  cos_results.writePacket<0>(0, c);

  Eigen::internal::psincos(x.packet<0>(4), s, c);
  sin_results.writePacket<0>(4, s);
  cos_results.writePacket<0>(4, c);

  std::cout << "Sin results: " << sin_results.transpose() << std::endl;
  std::cout << "Cos results: " << cos_results.transpose() << std::endl;

  for (int i = 0; i < VECTOR_SIZE; ++i) {
    // Validate sine results
    const float expected_sin = std::sin(x[i]);
    if (std::isnan(expected_sin)) {
      EXPECT_TRUE(std::isnan(sin_results[i]));
    } else {
      EXPECT_FLOAT_EQ(expected_sin, sin_results[i]);
    }

    // Validate cosine results
    const float expected_cos = std::cos(x[i]);
    if (std::isnan(expected_cos)) {
      EXPECT_TRUE(std::isnan(cos_results[i]));
    } else {
      EXPECT_FLOAT_EQ(expected_cos, cos_results[i]);
    }
  }
}

/**
 * @brief Test vectorized square root operation
 */
TEST_F(VectorizedMathTest, SquareRootOperation) {
  Eigen::ArrayXf eigen_results(VECTOR_SIZE);
  const auto &x = test_inputs;

  // Perform vectorized sqrt operations
  eigen_results.writePacket<0>(0, Eigen::internal::psqrt(x.packet<0>(0)));
  eigen_results.writePacket<0>(4, Eigen::internal::psqrt(x.packet<0>(4)));

  std::cout << "Sqrt results: " << eigen_results.transpose() << std::endl;

  for (int i = 0; i < VECTOR_SIZE; ++i) {
    const float expected = std::sqrt(x[i]);
    if (std::isnan(expected)) {
      EXPECT_TRUE(std::isnan(eigen_results[i]));
    } else {
      EXPECT_FLOAT_EQ(expected, eigen_results[i]);
    }
  }
}

#endif // EIGEN_VECTORIZE_NEON

/**
 * @brief Template test fixture for continuous distribution tests
 *
 * @tparam T Floating point type to test (float or double)
 */
template <typename T>
class ContinuousDistributionTest : public ::testing::Test {
protected:
  using MatrixType = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
  static constexpr int SAMPLES = 10000;
  static constexpr T TOLERANCE = 1e-2;
};

using FloatTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(ContinuousDistributionTest, FloatTypes);

/**
 * @brief Test balanced distribution properties
 */
TYPED_TEST(ContinuousDistributionTest, BalancedDistribution) {
  using MatrixType = typename TestFixture::MatrixType;
  // Test implementation would go here
  // Would include checks for mean, variance, and distribution shape
}
