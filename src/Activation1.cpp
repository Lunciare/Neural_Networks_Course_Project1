#include "Activation1.h"
#include <cassert> // Для assert
#include <cmath>   // Для exp, tanh
#include <utility> // Для std::move

namespace neural_network {

/**
 * @brief Конструктор функции активации
 * @param activation Функция активации (f(x))
 * @param derivative Производная функции активации (f'(x))
 */
ActivationFunction::ActivationFunction(Function activation, Function derivative)
    : activation_(std::move(activation)), derivative_(std::move(derivative)) {
  // Проверки можно добавить здесь, если требуется
}

/**
 * @brief Создает функцию активации ReLU (Rectified Linear Unit)
 * @return Объект ActivationFunction с ReLU
 *
 * ReLU(x) = max(0, x)
 * Производная: 1 при x > 0, иначе 0
 */
ActivationFunction ActivationFunction::ReLU() {
  return ActivationFunction(
      [](double x) {
        return x > 0 ? x : 0.0;
      }, // Более читаемая форма чем (x > 0)*x
      [](double x) { return x > 0 ? 1.0 : 0.0; });
}

/**
 * @brief Создает сигмоидную функцию активации
 * @return Объект ActivationFunction с сигмоидой
 *
 * Сигмоида: 1 / (1 + e^-x)
 * Производная: sigmoid(x)*(1 - sigmoid(x))
 */
ActivationFunction ActivationFunction::Sigmoid() {
  return ActivationFunction(
      [](double x) {
        // Добавляем 1.0 для явного указания типа
        return 1.0 / (1.0 + std::exp(-x));
      },
      [](double x) {
        const double s = 1.0 / (1.0 + std::exp(-x));
        return s * (1.0 - s);
      });
}

/**
 * @brief Создает тождественную функцию активации
 * @return Объект ActivationFunction с f(x) = x
 *
 * Тождественная функция: f(x) = x
 * Производная: f'(x) = 1
 */
ActivationFunction ActivationFunction::Identity() {
  return ActivationFunction(
      [](double x) { return x; },
      [](double) { return 1.0; } // Явно указываем возвращаемый тип
  );
}

/**
 * @brief Создает гиперболический тангенс в качестве функции активации
 * @return Объект ActivationFunction с tanh
 *
 * Tanh: (e^x - e^-x) / (e^x + e^-x)
 * Производная: 1 - tanh²(x)
 */
ActivationFunction ActivationFunction::Tanh() {
  return ActivationFunction([](double x) { return std::tanh(x); },
                            [](double x) {
                              const double t = std::tanh(x);
                              return 1.0 - t * t;
                            });
}

/**
 * @brief Вычисляет значение функции активации для скаляра
 * @param x Входное значение
 * @return Результат применения функции активации
 * @throws Может вызывать assert если функция не инициализирована
 */
double ActivationFunction::evaluate(double x) const {
  assert(activation_ && "Activation function not initialized");
  return activation_(x);
}

/**
 * @brief Вычисляет производную функции активации для скаляра
 * @param x Входное значение
 * @return Результат применения производной
 * @throws Может вызывать assert если функция не инициализирована
 */
double ActivationFunction::derEvaluate(double x) const {
  assert(derivative_ && "Derivative function not initialized");
  return derivative_(x);
}

/**
 * @brief Применяет функцию активации к каждому элементу матрицы
 * @param x Входная матрица
 * @return Матрица после применения функции активации
 * @throws Может вызывать assert если функция не инициализирована
 */
Matrix ActivationFunction::evaluate(const Matrix &x) const {
  assert(activation_ && "Activation function not initialized");
  return x.unaryExpr(activation_);
}

/**
 * @brief Применяет производную функции активации к каждому элементу матрицы
 * @param x Входная матрица
 * @return Матрица после применения производной
 * @throws Может вызывать assert если функция не инициализирована
 */
Matrix ActivationFunction::derEvaluate(const Matrix &x) const {
  assert(derivative_ && "Derivative function not initialized");
  return x.unaryExpr(derivative_);
}

} // namespace neural_network