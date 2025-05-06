#include "DataLoader.h"
#include <algorithm> // для std::min
#include <stdexcept> // для исключений

namespace neural_network {

/**
 * @brief Конструктор DataLoader
 * @param data Матрица данных (каждый столбец - отдельный образец)
 * @param labels Матрица меток (соответствует столбцам данных)
 * @param batch_size Размер батча
 * @param normalize_status Флаг нормализации данных
 * @throws std::invalid_argument Если данные и метки не согласованы
 */
DataLoader::DataLoader(const Matrix &data, const Matrix &labels,
                       Index batch_size, NormalizeStatus normalize_status)
    : data_(normalize_status == NormalizeStatus::Active ? normalizeData(data)
                                                        : data),
      labels_(labels), batch_size_(batch_size),
      num_batches_(calculateBatchCount(data.cols(), batch_size)) {
  validateInput(data, labels);
}

/**
 * @brief Проверяет согласованность входных данных
 */
void DataLoader::validateInput(const Matrix &data, const Matrix &labels) const {
  if (data.cols() != labels.cols()) {
    throw std::invalid_argument(
        "Data and labels must have same number of columns");
  }
  if (batch_size_ <= 0) {
    throw std::invalid_argument("Batch size must be positive");
  }
}

/**
 * @brief Вычисляет количество батчей
 */
Index DataLoader::calculateBatchCount(Index total_samples,
                                      Index batch_size) const {
  return (total_samples + batch_size - 1) / batch_size;
}

/**
 * @brief Нормализует данные в диапазон [0, 1]
 * @throws std::runtime_error Если все значения одинаковы
 */
Matrix DataLoader::normalizeData(Matrix data) { const auto min_val = data }

} // namespace neural_network