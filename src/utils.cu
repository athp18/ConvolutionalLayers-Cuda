// utils.cu
#include "utils.h"
#include <fstream>
#include <iostream>
#include <cmath>

std::vector<float> loadKernel(const std::string& filename, int& kernel_width, int& kernel_height)
{
    std::ifstream infile(filename);
    if (!infile)
    {
        std::cerr << "Failed to open kernel file: " << filename << std::endl;
        exit(1);
    }

    infile >> kernel_width >> kernel_height;
    std::vector<float> kernel(kernel_width * kernel_height);
    for(auto &val : kernel)
    {
        infile >> val;
    }

    return kernel;
}

bool verifyOutput(const std::vector<float>& output, float expected_value, int image_width, int image_height)
{
    for(int i = 0; i < image_height; ++i)
    {
        for(int j = 0; j < image_width; ++j)
        {
            float val = output[i * image_width + j];
            if (fabs(val - expected_value) > 1e-5)
            {
                std::cerr << "Mismatch at (" << i << ", " << j << "): " << val << " != " << expected_value << std::endl;
                return false;
            }
        }
    }
    return true;
}