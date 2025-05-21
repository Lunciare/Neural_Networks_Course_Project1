#include "MNISTLoader.h"
#include <cstdint>
#include <fstream>

namespace neural_network
{

// Reads a 32-bit unsigned integer in big-endian from file
static uint32_t readUint32BE(std::ifstream& in)
{
    unsigned char bytes[4];
    in.read(reinterpret_cast<char*>(bytes), 4);
    return (uint32_t(bytes[0]) << 24) | (uint32_t(bytes[1]) << 16) |
           (uint32_t(bytes[2]) << 8) | uint32_t(bytes[3]);
}

bool loadMNIST(const std::string& image_file, const std::string& label_file,
               std::vector<Eigen::VectorXd>& images, std::vector<int>& labels)
{
    std::ifstream img_f(image_file, std::ios::binary),
            lbl_f(label_file, std::ios::binary);
    if (!img_f.is_open() || !lbl_f.is_open())
        return false;
    uint32_t magic_images = readUint32BE(img_f), num_images = readUint32BE(img_f);
    uint32_t rows = readUint32BE(img_f), cols = readUint32BE(img_f);
    uint32_t magic_labels = readUint32BE(lbl_f), num_labels = readUint32BE(lbl_f);
    if (magic_images != 2051 || magic_labels != 2049 || num_images != num_labels)
        return false;
    images.clear();
    labels.clear();
    images.reserve(num_images);
    labels.reserve(num_images);
    size_t img_size = rows * cols;
    std::vector<unsigned char> buffer(img_size);
    for (uint32_t i = 0; i < num_images; ++i)
    {
        img_f.read(reinterpret_cast<char*>(buffer.data()), img_size);
        unsigned char label = 0;
        lbl_f.read(reinterpret_cast<char*>(&label), 1);
        Eigen::VectorXd vec(img_size);
        for (size_t j = 0; j < img_size; ++j)
            vec[j] = buffer[j] / 255.0;
        images.push_back(vec);
        labels.push_back(int(label));
    }
    return true;
}

}// namespace neural_network
