// main.cu
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include "conv2D.h"
#include "utils.h"

#define IMAGE_WIDTH  1024
#define IMAGE_HEIGHT 1024
#define KERNEL_WIDTH 3
#define KERNEL_HEIGHT 3
#define TILE_WIDTH 16

int main()
{
    // Initialize image and kernel dimensions
    const int image_size = IMAGE_WIDTH * IMAGE_HEIGHT;
    const int kernel_size = KERNEL_WIDTH * KERNEL_HEIGHT;

    // Allocate host memory
    std::vector<float> h_input(image_size, 1.0f); // Example: initialize all pixels to 1
    std::vector<float> h_kernel(kernel_size, 1.0f / kernel_size); // Example: averaging filter
    std::vector<float> h_output(image_size, 0.0f);

    // Device pointers
    float *d_input = nullptr, *d_kernel = nullptr, *d_output = nullptr;

    // Allocate device memory
    cudaMalloc((void**)&d_input, image_size * sizeof(float));
    cudaCheckErrors("cudaMalloc d_input failed");

    cudaMalloc((void**)&d_kernel, kernel_size * sizeof(float));
    cudaCheckErrors("cudaMalloc d_kernel failed");

    cudaMalloc((void**)&d_output, image_size * sizeof(float));
    cudaCheckErrors("cudaMalloc d_output failed");

    // Copy data from host to device
    cudaMemcpy(d_input, h_input.data(), image_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy h_input to d_input failed");

    cudaMemcpy(d_kernel, h_kernel.data(), kernel_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy h_kernel to d_kernel failed");

    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((IMAGE_WIDTH + block.x - 1) / block.x, (IMAGE_HEIGHT + block.y - 1) / block.y);

    // compute shared memory size
    int shared_mem_size = (TILE_WIDTH + 2 * (KERNEL_WIDTH / 2)) * (TILE_WIDTH + 2 * (KERNEL_HEIGHT / 2)) * sizeof(float);

    // Launch the kernel
    launchConv2DOptimizedKernel(d_input, d_output,
                                IMAGE_WIDTH, IMAGE_HEIGHT,
                                KERNEL_WIDTH, KERNEL_HEIGHT,
                                grid, block, shared_mem_size);
    cudaCheckErrors("Optimized Kernel launch failed");

    /*
    launchConv2DBasicKernel(d_input, d_kernel, d_output,
                            IMAGE_WIDTH, IMAGE_HEIGHT,
                            KERNEL_WIDTH, KERNEL_HEIGHT,
                            grid, block, shared_mem_size);
    cudaCheckErrors("Basic Kernel launch failed");
    */

    // Wait for GPU to finish
    cudaDeviceSynchronize();
    cudaCheckErrors("cudaDeviceSynchronize failed");

    // Copy the result back to host
    cudaMemcpy(h_output.data(), d_output, image_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpy d_output to h_output failed");

    // (Optional) Verify the result
    bool correct = verifyOutput(h_output, 1.0f, IMAGE_WIDTH, IMAGE_HEIGHT);

    if (correct)
    {
        std::cout << "Convolution successful!" << std::endl;
    }
    else
    {
        std::cout << "Convolution failed!" << std::endl;
    }

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);

    return 0;
}
