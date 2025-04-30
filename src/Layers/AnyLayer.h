#pragma once

#include <tuple>
#include <utility>
#include <variant>

#include "DenseLayer.h"
#include "DropoutLayer.h"

namespace neural_network {
class AnyLayer {
public:
  using LayerType = std::variant<DenseLayer, DropoutLayer>;

  static AnyLayer createDenseLayer(In in_size, Out out_size,
                                   ActivationFunction f, Optimizer opt);
  static AnyLayer createDropoutLayer(In in_size, Out out_size, double rate);

  Matrix evaluate(const Matrix &input) const;

  template <typename Cache>
  Matrix cachedEvaluate(const Matrix &input, Cache &cache) const {
    return std::visit(
        [&](const auto &l) -> Matrix {
          using LayerT = std::decay_t<decltype(l)>;
          if constexpr (std::is_same_v<LayerT, DropoutLayer> &&
                        std::is_same_v<Cache,
                                       typename DropoutLayer::DropoutCache>) {
            auto ptr =
                static_cast<Matrix (LayerT::*)(const Matrix &, Cache &) const>(
                    &LayerT::evaluate);
            return (l.*ptr)(input, cache);
          } else {
            auto ptr = static_cast<Matrix (LayerT::*)(const Matrix &) const>(
                &LayerT::evaluate);
            (void)cache;
            return (l.*ptr)(input);
          }
        },
        layer_);
  }

  Matrix getBackpropError(const Matrix &a, const Matrix &z,
                          const Matrix &b) const;

  template <typename Cache>
  Matrix cachedBackpropError(const Matrix &a, const Matrix &z, const Matrix &b,
                             const Cache &cache) const {
    return std::visit(
        [&](const auto &l) -> Matrix {
          return callBackpropErrorWithCache(l, a, z, b, cache);
        },
        layer_);
  }

  Matrix getGradW(const Matrix &a, const Matrix &z, const Matrix &b) const;
  Matrix getGradB(const Matrix &a, const Matrix &z, const Matrix &b) const;
  bool hasWeights() const;

  void updateW(const Matrix &grad_diff, Matrix &memory, int time_step);
  void updateB(const Vector &grad_diff, Vector &memory, int time_step);

  Index getInputSize() const;
  Index getOutputSize() const;

  const Matrix &getWeights() const;
  const Vector &getBiases() const;

private:
  explicit AnyLayer(LayerType layer);
  LayerType layer_;

  template <typename L, typename Cache>
  static auto callBackpropErrorWithCache(const L &layer, const Matrix &a,
                                         const Matrix &z, const Matrix &b,
                                         const Cache &cache)
      -> decltype(layer.getBackpropError(a, z, b, cache)) {
    return layer.getBackpropError(a, z, b, cache);
  }

  template <typename L, typename Cache>
  static auto callBackpropErrorWithCache(const L &layer, const Matrix &a,
                                         const Matrix &z, const Matrix &b,
                                         const Cache &cache)
      -> decltype(layer.getBackpropError(a, z, b)) {
    (void)cache;
    return layer.getBackpropError(a, z, b);
  }
};

} // namespace neural_network
