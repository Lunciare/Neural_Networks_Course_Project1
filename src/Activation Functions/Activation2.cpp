#pragma once

#include "Math.h"     // Для работы с матрицами
#include <cmath>      // Для математических функций
#include <functional> // Для std::function
#include <string>     // Для строковых исключений

namespace neural_network {

/**
 * @class ActivationFunction
 * @brief Класс, представляющий функцию активации и её производную
 *
 * Содержит как саму функцию активации, так и её производную, что важно
 * для алгоритмов обучения нейронных сетей, таких как обратное распространение
 * ошибки.
 */
class ActivationFunction {
public:
  /// Тип для функций активации (принимает double, возвращает double)
  using Function = std::function<double(double)>;

  /**
   * @brief Конструктор функции активации
   * @param activation Функция активации f(x)
   * @param derivative Производная функция f'(x)
   * @throws std::invalid_argument Если любая из функций не задана
   */
  ActivationFunction(Function activation, Function derivative)
      : activation_(std::move(activation)), derivative_(std::move(derivative)) {
    if (!activation_ || !derivative_) {
      throw std::invalid_argument(
          "Activation and derivative functions must be provided");
    }
  }

  // Статические фабричные методы для стандартных функций активации

  /**
   * @brief Создаёт функцию активации ReLU (Rectified Linear Unit)
   * @return ReLU: f(x) = max(0, x), f'(x) = 1 if x > 0 else 0
   */
  static ActivationFunction ReLU() {
    return ActivationFunction([](double x) { return std::max(0.0, x); },
                              [](double x) { return x > 0 ? 1.0 : 0.0; });
  }

  /**
   * @brief Создаёт сигмоидную функцию активации
   * @return Sigmoid: f(x) = 1 / (1 + exp(-x)), f'(x) = f(x)*(1 - f(x))
   */
  static ActivationFunction Sigmoid() {
    return ActivationFunction(
        [](double x) { return 1.0 / (1.0 + std::exp(-x)); },
        [](double x) {
          double s = 1.0 / (1.0 + std::exp(-x));
          return s * (1.0 - s);
        });
  }

  /**
   * @brief Создаёт тождественную функцию активации
   * @return Identity: f(x) = x, f'(x) = 1
   */
  static ActivationFunction Identity() {
    return ActivationFunction([](double x) { return x; },
                              [](double) { return 1.0; });
  }

  /**
   * @brief Создаёт функцию активации на основе гиперболического тангенса
   * @return Tanh: f(x) = tanh(x), f'(x) = 1 - tanh²(x)
   */
  static ActivationFunction Tanh() {
    return ActivationFunction([](double x) { return std::tanh(x); },
                              [](double x) {
                                double t = std::tanh(x);
                                return 1.0 - t * t;
                              });
  }

  // Методы для применения функций

  /**
   * @brief Применяет функцию активации к скалярному значению
   * @param x Входное значение
   * @return Результат применения функции активации
   */
  double evaluate(double x) const { return activation_(x); }

  /**
   * @brief Применяет производную функции активации к скалярному значению
   * @param x Входное значение
   * @return Результат применения производной
   */
  double derEvaluate(double x) const { return derivative_(x); }

  /**
   * @brief Применяет функцию активации к каждому элементу матрицы
   * @param x Входная матрица
   * @return Матрица после применения функции активации
   */
  Matrix evaluate(const Matrix &x) const { return x.unaryExpr(activation_); }

  /**
   * @brief Применяет производную функции активации к каждому элементу матрицы
   * @param x Входная матрица
   * @return Матрица после применения производной
   */
  Matrix derEvaluate(const Matrix &x) const { return x.unaryExpr(derivative_); }

private:
  Function activation_; ///< Функция активации f(x)
  Function derivative_; ///< Производная функции активации f'(x)
};

} // namespace neural_network