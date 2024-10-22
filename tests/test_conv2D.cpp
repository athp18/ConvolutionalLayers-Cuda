// test_conv2D.cpp
#include <gtest/gtest.h>
#include "conv2D.h"
#include "utils.h"
#include <vector>

TEST(ConvolutionTest, BasicTest)
{
    // Define small image and kernel for testing
    const int image_width = 5;
    const int image_height = 5;
    const int kernel_width = 3;
    const int kernel_height = 3;

    std::vector<float> h_input = {
        1, 1, 1, 1, 1,
        1, 2, 2, 2, 1,
        1, 2, 3, 2, 1,
        1, 2, 2, 2, 1,
        1, 1, 1, 1, 1
    };

    std::vector<float> h_kernel = {
        1, 1, 1,
        1, 1, 1,
        1, 1, 1
    };

    std::vector<float> h_output(25, 0.0f);

    // Device pointers
    float *d_input = nullptr, *d_kernel = nullptr, *d_output_device = nullptr;

    // Allocate device memory
    cudaMalloc((void**)&d_input, h_input.size() * sizeof(float));
    cudaCheckErrors("cudaMalloc d_input failed");

    cudaMalloc((void**)&d_kernel, h_kernel.size() * sizeof(float));
    cudaCheckErrors("cudaMalloc d_kernel failed");

    cudaMalloc((void**)&d_output_device, h_output.size() * sizeof(float));
    cudaCheckErrors("cudaMalloc d_output_device failed");

    // Copy data to device
    cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel.data(), h_kernel.size() * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 block(16, 16);
    dim3 grid((image_width + block.x - 1) / block.x, (image_height + block.y - 1) / block.y);

    // Calculate shared memory size
    int shared_mem_size = (16 + 2 * (kernel_width / 2)) * (16 + 2 * (kernel_height / 2)) * sizeof(float);

    // Launch basic kernel
    launchConv2DBasicKernel(d_input, d_kernel, d_output_device,
                            image_width, image_height,
                            kernel_width, kernel_height,
                            grid, block, shared_mem_size);
    cudaCheckErrors("Basic Kernel launch failed");

    // Synchronize
    cudaDeviceSynchronize();
    cudaCheckErrors("cudaDeviceSynchronize failed");

    // Copy result back to host
    cudaMemcpy(h_output.data(), d_output_device, h_output.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpy d_output to h_output failed");

    // Define expected output manually or using a CPU implementation
    std::vector<float> expected_output = {
        6, 9, 12, 9, 6,
        9, 15, 18, 15, 9,
        12, 18, 21, 18, 12,
        9, 15, 18, 15, 9,
        6, 9, 12, 9, 6
    };

    // Verify basic kernel
    for(int i = 0; i < h_output.size(); ++i)
    {
        EXPECT_FLOAT_EQ(h_output[i], expected_output[i]);
    }

    // Clean up
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output_device);
}

TEST(ConvolutionTest, OptimizedTest)
{
    // Define small image and kernel for testing
    const int image_width = 5;
    const int image_height = 5;
    const int kernel_width = 3;
    const int kernel_height = 3;

    std::vector<float> h_input = {
        1, 1, 1, 1, 1,
        1, 2, 2, 2, 1,
        1, 2, 3, 2, 1,
        1, 2, 2, 2, 1,
        1, 1, 1, 1, 1
    };

    std::vector<float> h_kernel = {
        1, 1, 1,
        1, 1, 1,
        1, 1, 1
    };

    std::vector<float> h_output(25, 0.0f);

    // Device pointers
    float *d_input = nullptr, *d_output_device = nullptr;

    // Allocate device memory
    cudaMalloc((void**)&d_input, h_input.size() * sizeof(float));
    cudaCheckErrors("cudaMalloc d_input failed");

    cudaMalloc((void**)&d_output_device, h_output.size() * sizeof(float));
    cudaCheckErrors("cudaMalloc d_output_device failed");

    // Copy input data to device
    cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy h_input to d_input failed");

    // Copy convolution kernel to constant memory
    cudaMemcpyToSymbol(d_kernel_const, h_kernel.data(), kernel_width * kernel_height * sizeof(float));
    cudaCheckErrors("cudaMemcpy to constant memory d_kernel_const failed");

    // Define block and grid dimensions
    dim3 block(16, 16);
    dim3 grid((image_width + block.x - 1) / block.x, (image_height + block.y - 1) / block.y);

    // Calculate shared memory size
    int shared_mem_size = (16 + 2 * (kernel_width / 2)) * (16 + 2 * (kernel_height / 2)) * sizeof(float);

    // Launch optimized kernel
    launchConv2DOptimizedKernel(d_input, d_output_device,
                                image_width, image_height,
                                kernel_width, kernel_height,
                                grid, block, shared_mem_size);
    cudaCheckErrors("Optimized Kernel launch failed");

    // Synchronize
    cudaDeviceSynchronize();
    cudaCheckErrors("cudaDeviceSynchronize failed");

    // Copy result back to host
    cudaMemcpy(h_output.data(), d_output_device, h_output.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpy d_output to h_output failed");

    // Define expected output manually or using a CPU implementation
    std::vector<float> expected_output = {
        6, 9, 12, 9, 6,
        9, 15, 18, 15, 9,
        12, 18, 21, 18, 12,
        9, 15, 18, 15, 9,
        6, 9, 12, 9, 6
    };

    // Verify optimized kernel
    for(int i = 0; i < h_output.size(); ++i)
    {
        EXPECT_FLOAT_EQ(h_output[i], expected_output[i]);
    }

    // Clean up
    cudaFree(d_input);
    cudaFree(d_output_device);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}