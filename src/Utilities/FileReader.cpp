#include "Utilities/FileReader.h"

namespace neural_network {

FileReader::FileReader(const std::string &filename) : file_(filename) {}

FileReader::~FileReader() {
  if (file_.is_open())
    file_.close();
}

bool FileReader::good() const { return file_.good(); }

} // namespace neural_network
