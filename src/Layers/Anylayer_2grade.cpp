#include "AnyLayer.h"
#include <type_traits>
#include <variant>

namespace neural_network {

  // ======================================================
  // Вспомогательные метафункции для проверки интерфейсов
  // ======================================================

  namespace internal {

    /**
     * @brief Проверяет наличие метода getWeights у типа
     */
    template<typename T, typename = void> struct HasGetWeights
        : std::false_type {};

    template<typename T> struct HasGetWeights<
        T, std::void_t<decltype(std::declval<T>().getWeights())>>
        : std::true_type {};

    /**
     * @brief Проверяет наличие метода getBiases у типа
     */
    template<typename T, typename = void> struct HasGetBiases
        : std::false_type {};

    template<typename T> struct HasGetBiases<
        T, std::void_t<decltype(std::declval<T>().getBiases())>>
        : std::true_type {};

  } // namespace internal

  // ======================================================
  // Реализация методов AnyLayer
  // ======================================================

  AnyLayer::AnyLayer(LayerType layer) : layer_(std::move(layer)) {}

  AnyLayer AnyLayer::createDenseLayer(In in_size, Out out_size,
                                      ActivationFunction f, Optimizer opt) {
    return AnyLayer(
        DenseLayer(in_size, out_size, std::move(f), std::move(opt)));
  }

  AnyLayer AnyLayer::createDropoutLayer(In in_size, Out out_size, double rate) {
    return AnyLayer(DropoutLayer(in_size, out_size, rate));
  }

  Matrix AnyLayer::evaluate(const Matrix &input) const {
    return std::visit(
        [&input](const auto &layer) { return layer.evaluate(input); }, layer_);
  }

  Matrix AnyLayer::getBackpropError(const Matrix &activation,
                                    const Matrix &weighted_input,
                                    const Matrix &error) const {
    return std::visit(
        [&](const auto &layer) -> Matrix {
          if constexpr (requires {
                          layer.getBackpropError(activation, weighted_input,
                                                 error);
                        }) {
            return layer.getBackpropError(activation, weighted_input, error);
          } else {
            return activation; // По умолчанию возвращаем входной сигнал
          }
        },
        layer_);
  }

  Matrix AnyLayer::getGradW(const Matrix &activation,
                            const Matrix &weighted_input, const Matrix &error)
      const {
    return std::visit(
        [&](const auto &layer) -> Matrix {
          if constexpr (std::is_same_v<std::decay_t<decltype(layer)>,
                                       DropoutLayer>) {
            return {}; // Dropout layer не имеет весов
          } else if constexpr (requires {
                                 layer.getGradW(activation, weighted_input,
                                                error);
                               }) {
            return layer.getGradW(activation, weighted_input, error);
          }
          return {}; // Пустая матрица по умолчанию
        },
        layer_);
  }

  Matrix AnyLayer::getGradB(const Matrix &activation,
                            const Matrix &weighted_input, const Matrix &error)
      const {
    return std::visit(
        [&](const auto &layer) -> Matrix {
          if constexpr (std::is_same_v<std::decay_t<decltype(layer)>,
                                       DropoutLayer>) {
            return {}; // Dropout layer не имеет смещений
          } else if constexpr (requires {
                                 layer.getGradB(activation, weighted_input,
                                                error);
                               }) {
            return layer.getGradB(activation, weighted_input, error);
          }
          return {}; // Пустая матрица по умолчанию
        },
        layer_);
  }

  bool AnyLayer::hasWeights() const {
    return std::visit(
        [](const auto &layer) {
          if constexpr (std::is_same_v<std::decay_t<decltype(layer)>,
                                       DropoutLayer>) {
            return false;
          } else {
            return internal::HasGetWeights<
                std::decay_t<decltype(layer)>>::value;
          }
        },
        layer_);
  }

  void AnyLayer::updateW(const Matrix &grad_diff, Matrix &memory,
                         int time_step) {
    std::visit(
        [&](auto &layer) {
          if constexpr (requires {
                          layer.updateW(grad_diff, memory, time_step);
                        }) {
            layer.updateW(grad_diff, memory, time_step);
          }
        },
        layer_);
  }

  void AnyLayer::updateB(const Vector &grad_diff, Vector &memory,
                         int time_step) {
    std::visit(
        [&](auto &layer) {
          if constexpr (requires {
                          layer.updateB(grad_diff, memory, time_step);
                        }) {
            layer.updateB(grad_diff, memory, time_step);
          }
        },
        layer_);
  }

  const Matrix &AnyLayer::getWeights() const {
    static const Matrix empty_matrix = Matrix::Zero(0, 0);

    return std::visit(
        [](const auto &layer) -> const Matrix & {
          if constexpr (internal::HasGetWeights<
                            std::decay_t<decltype(layer)>>::value) {
            return layer.getWeights();
          } else {
            return empty_matrix;
          }
        },
        layer_);
  }

  const Vector &AnyLayer::getBiases() const {
    static const Vector empty_vector = Vector::Zero(0);

    return std::visit(
        [](const auto &layer) -> const Vector & {
          if constexpr (internal::HasGetBiases<
                            std::decay_t<decltype(layer)>>::value) {
            return layer.getBiases();
          } else {
            return empty_vector;
          }
        },
        layer_);
  }

  Index AnyLayer::getInputSize() const {
    return std::visit(
        [](const auto &layer) -> Index {
          if constexpr (requires { layer.getInputSize(); }) {
            return layer.getInputSize();
          }
          return 0;
        },
        layer_);
  }

  Index AnyLayer::getOutputSize() const {
    return std::visit(
        [](const auto &layer) -> Index {
          if constexpr (requires { layer.getOutputSize(); }) {
            return layer.getOutputSize();
          }
          return 0;
        },
        layer_);
  }

} // namespace neural_network