#ifndef MODEL_H
#define MODEL_H

#include "AnyLayer.h"
#include "LossFunction.h"
#include "Optimizer.h"
#include <string>
#include <vector>

namespace neural_network {

class Model {
public:
  Model();
  explicit Model(const std::string &model_name);

  void add(AnyLayer *layer);
  void compile(LossFunction *lossFunction, Optimizer *opt);
  void train(const std::vector<std::vector<double>> &X,
             const std::vector<std::vector<double>> &y, int epochs,
             int batch_size);

  std::vector<std::vector<double>>
  predict(const std::vector<std::vector<double>> &X);
  void summary() const;

private:
  std::vector<AnyLayer *> layers;
  LossFunction *loss;
  Optimizer *optimizer;
  std::string name;
};

} // namespace neural_network

#endif // MODEL_H
