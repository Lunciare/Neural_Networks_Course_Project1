#include "Utilities/FileWriter.h"

namespace neural_network {

FileWriter::FileWriter(const std::string &filename) : file_(filename) {}

FileWriter::~FileWriter() {
  if (file_.is_open())
    file_.close();
}

bool FileWriter::good() const { return file_.good(); }

} // namespace neural_network
