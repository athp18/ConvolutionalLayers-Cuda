/*
Kernel reader to process a text file
*/
#include <fstream>
#include <vector>
#include <stdexcept>
#include <sstream>
#include <iomanip>
#include "cuda_runtime.h"

class KernelReader {
public:
    struct Kernel {
        std::vector<float> values;
        int width;
        int height;
    };

    static Kernel read_file(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open kernel file: " + filename);
        }

        Kernel kernel;
        std::string firstLine;
        
        if (!std::getline(file, firstLine)) {
            throw std::runtime_error("Failed to read kernel dimensions");
        }

        std::istringstream iss(firstLine);
        if (!(iss >> kernel.width >> kernel.height)) {
            throw std::runtime_error("Invalid kernel dimensions format");
        }

        if (kernel.width <= 0 || kernel.height <= 0) {
            throw std::runtime_error("Invalid kernel dimensions: must be positive");
        }

        kernel.values.reserve(kernel.width * kernel.height);

        float value;
        for (int i = 0; i < kernel.height; ++i) {
            std::string line;
            if (!std::getline(file, line)) {
                throw std::runtime_error("Insufficient kernel data rows");
            }

            std::istringstream rowStream(line);
            int valuesRead = 0;
            
            while (rowStream >> value) {
                if (valuesRead >= kernel.width) {
                    throw std::runtime_error("Too many values in row " + std::to_string(i + 1));
                }
                kernel.values.push_back(value);
                valuesRead++;
            }

            if (valuesRead < kernel.width) {
                throw std::runtime_error("Insufficient values in row " + std::to_string(i + 1));
            }
        }

        std::string extraLine;
        if (std::getline(file, extraLine)) {
            // Trim whitespace
            extraLine.erase(0, extraLine.find_first_not_of(" \t\n\r\f\v"));
            extraLine.erase(extraLine.find_last_not_of(" \t\n\r\f\v") + 1);
            
            if (!extraLine.empty()) {
                throw std::runtime_error("Extra data found after kernel matrix");
            }
        }

        return kernel;
    }

    static void printKernel(const Kernel& kernel) {
        std::cout << "Kernel dimensions: " << kernel.width << "x" << kernel.height << "\n\n";
        
        for (int i = 0; i < kernel.height; ++i) {
            for (int j = 0; j < kernel.width; ++j) {
                std::cout << std::setw(8) << std::fixed << std::setprecision(4) 
                         << kernel.values[i * kernel.width + j] << " ";
            }
            std::cout << "\n";
        }
    }

    static bool validate(const Kernel& kernel) {
        if (kernel.values.size() != static_cast<size_t>(kernel.width * kernel.height)) {
            return false;
        }
        return true;
    }

    // helper function to copy kernel to GPU constant memory
    static void copyToConstantMemory(const Kernel& kernel) {
        if (kernel.width * kernel.height > MAX_KERNEL_WIDTH * MAX_KERNEL_HEIGHT) {
            throw std::runtime_error("Kernel size exceeds maximum supported dimensions");
        }

        cudaError_t err = cudaMemcpyToSymbol(d_kernel_const, 
                                            kernel.values.data(),
                                            kernel.values.size() * sizeof(float));
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to copy kernel to constant memory: " + 
                                   std::string(cudaGetErrorString(err)));
        }
    }
};
