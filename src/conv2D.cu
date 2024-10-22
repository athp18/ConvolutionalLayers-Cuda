// conv2D.cu
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "conv2D.h"

// Basic convolution kernel
__global__ void conv2DBasic(const float* __restrict__ d_input,
                            const float* __restrict__ d_kernel,
                            float* d_output,
                            int image_width,
                            int image_height,
                            int kernel_width,
                            int kernel_height)
{
    // Calculate the row and column index of the output element
    int out_row = blockIdx.y * blockDim.y + threadIdx.y;
    int out_col = blockIdx.x * blockDim.x + threadIdx.x;

    // Compute the half size of the kernel
    int k_half_width = kernel_width / 2;
    int k_half_height = kernel_height / 2;

    float sum = 0.0f;

    // Iterate over the kernel
    for (int i = -k_half_height; i <= k_half_height; ++i)
    {
        for (int j = -k_half_width; j <= k_half_width; ++j)
        {
            int in_row = out_row + i;
            int in_col = out_col + j;

            // Handle boundary conditions by zero-padding
            if (in_row >= 0 && in_row < image_height && in_col >= 0 && in_col < image_width)
            {
                float input_val = d_input[in_row * image_width + in_col];
                float kernel_val = d_kernel[(i + k_half_height) * kernel_width + (j + k_half_width)];
                sum += input_val * kernel_val;
            }
        }
    }

    // Write the output
    if (out_row < image_height && out_col < image_width)
    {
        d_output[out_row * image_width + out_col] = sum;
    }
}

// Function to launch the basic convolution kernel
void launchConv2DBasicKernel(const float* d_input,
                              const float* d_kernel,
                              float* d_output,
                              int image_width,
                              int image_height,
                              int kernel_width,
                              int kernel_height,
                              dim3 grid,
                              dim3 block,
                              size_t shared_mem_size)
{
    conv2DBasic<<<grid, block>>>(d_input, d_kernel, d_output,
                                 image_width, image_height,
                                 kernel_width, kernel_height);
}