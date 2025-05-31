#pragma once

#include "Layers/Layer.h"
#include "LossFunctions/LossFunction.h"
#include <functional>
#include <initializer_list>
#include <vector>

namespace neural_network {

class Model {
public:
  // Конструктор: принимает список размеров слоёв и список функций активации
  Model(std::initializer_list<size_t> layer_sizes,
        std::initializer_list<ActivationFunction::Type> activations);

  // Прямой проход (inference). Теперь НЕ const, чтобы мы могли сохранять
  // внутренние кэши слоя
  Vector forward(const Vector &input);

  // Одна итерация обучения:
  //   1) Засетить кэш в каждом слое для заданного Optimizer
  //   2) Запустить forwardTrain(...) → получить градиент потерь на выходе
  //   3) Вызвать backward(...) → обновить веса
  //   4) Освободить кэш у каждого слоя
  void trainStep(
      const Vector &x, const Vector &y,
      const std::function<Vector(const Vector &, const Vector &)> &lossGrad,
      Optimizer &optimizer);

  // Тренировка в цикле по всем {xs, ys} заданное количество эпох
  void train(const std::vector<Vector> &xs, const std::vector<Vector> &ys,
             int epochs, LossFunction loss, Optimizer &optimizer);

  // Доступ к слоям (например, чтобы посмотреть веса/смещения)
  const std::vector<Layer> &layers() const;

  // ============ Новые декларации шаблонных методов read/write ============
  // Объявляем, что у Model есть методы .read(in) и .write(out),
  // чтобы Model.cpp мог их реализовать.
  template <class Reader> void read(Reader &in);

  template <class Writer> void write(Writer &out) const;
  // ========================================================================

private:
  std::vector<Layer> layers_;

  // Прямой проход при обучении (не const), сохраняет все промежуточные
  // активации.
  std::vector<Vector> forwardTrain(const Vector &x);

  // Обратный проход: из входного градиента делает optimizer.update(...) по
  // слоям
  void backward(const Vector &grad, const Optimizer &opt);
};

} // namespace neural_network
