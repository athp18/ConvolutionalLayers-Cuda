// utils.h
#ifndef UTILS_H
#define UTILS_H

#include <cuda_runtime.h>
#include <vector>
#include <string>

#define cudaCheckErrors(msg)                                \
    do                                                      \
    {                                                       \
        cudaError_t __err = cudaGetLastError();             \
        if (__err != cudaSuccess)                           \
        {                                                   \
            std::cerr << "Fatal error: " << msg << std::endl; \
            std::cerr << "Error code: " << cudaGetErrorString(__err) << std::endl; \
            exit(1);                                        \
        }                                                   \
    } while (0)

std::vector<float> loadKernel(const std::string& filename, int& kernel_width, int& kernel_height);
bool verifyOutput(const std::vector<float>& output, float expected_value, int image_width, int image_height);

#endif // UTILS_H