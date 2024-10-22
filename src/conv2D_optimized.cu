// conv2D_optimized.cu
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "conv2D.h"

// Define maximum kernel size for constant memory
#define MAX_KERNEL_WIDTH 7
#define MAX_KERNEL_HEIGHT 7

// Declare the convolution kernel in constant memory
__constant__ float d_kernel_const[MAX_KERNEL_WIDTH * MAX_KERNEL_HEIGHT];

// Optimized CUDA kernel with enhanced shared memory usage
__global__ void conv2DOptimized(const float* __restrict__ d_input,
                                float* d_output,
                                int image_width,
                                int image_height,
                                int kernel_width,
                                int kernel_height)
{
    // Compute the half sizes
    int k_half_width = kernel_width / 2;
    int k_half_height = kernel_height / 2;

    // Compute the global row and column indices of the output element
    int out_row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int out_col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    // Shared memory dimensions (tile + halo)
    int shared_width = TILE_WIDTH + 2 * k_half_width;
    int shared_height = TILE_WIDTH + 2 * k_half_height;

    // Allocate shared memory with padding to prevent bank conflicts
    extern __shared__ float shared_mem[];

    // Calculate the indices for loading data into shared memory
    int in_row = out_row - k_half_height;
    int in_col = out_col - k_half_width;

    // Load data into shared memory with boundary checks
    if (in_row >= 0 && in_row < image_height && in_col >= 0 && in_col < image_width)
    {
        shared_mem[threadIdx.y * shared_width + threadIdx.x] = d_input[in_row * image_width + in_col];
    }
    else
    {
        shared_mem[threadIdx.y * shared_width + threadIdx.x] = 0.0f;
    }

    // Load the halo regions
    // Load top halo
    if (threadIdx.y < k_half_height)
    {
        int halo_row = in_row - k_half_height;
        int halo_col = in_col;
        if (halo_row >= 0 && halo_col >= 0 && halo_col < image_width)
        {
            shared_mem[(threadIdx.y - k_half_height) * shared_width + threadIdx.x] = d_input[halo_row * image_width + halo_col];
        }
        else
        {
            shared_mem[(threadIdx.y - k_half_height) * shared_width + threadIdx.x] = 0.0f;
        }
    }

    // Load bottom halo
    if (threadIdx.y >= TILE_WIDTH - k_half_height)
    {
        int halo_row = in_row + TILE_WIDTH + k_half_height;
        int halo_col = in_col;
        if (halo_row >= 0 && halo_row < image_height && halo_col >= 0 && halo_col < image_width)
        {
            shared_mem[(threadIdx.y + k_half_height) * shared_width + threadIdx.x] = d_input[halo_row * image_width + halo_col];
        }
        else
        {
            shared_mem[(threadIdx.y + k_half_height) * shared_width + threadIdx.x] = 0.0f;
        }
    }

    // Load left halo
    if (threadIdx.x < k_half_width)
    {
        int halo_row = in_row;
        int halo_col = in_col - k_half_width;
        if (halo_row >= 0 && halo_row < image_height && halo_col >= 0 && halo_col < image_width)
        {
            shared_mem[threadIdx.y * shared_width + threadIdx.x - k_half_width] = d_input[halo_row * image_width + halo_col];
        }
        else
        {
            shared_mem[threadIdx.y * shared_width + threadIdx.x - k_half_width] = 0.0f;
        }
    }

    // Load right halo
    if (threadIdx.x >= TILE_WIDTH - k_half_width)
    {
        int halo_row = in_row;
        int halo_col = in_col + TILE_WIDTH + k_half_width;
        if (halo_row >= 0 && halo_row < image_height && halo_col >= 0 && halo_col < image_width)
        {
            shared_mem[threadIdx.y * shared_width + threadIdx.x + k_half_width] = d_input[halo_row * image_width + halo_col];
        }
        else
        {
            shared_mem[threadIdx.y * shared_width + threadIdx.x + k_half_width] = 0.0f;
        }
    }

    // Ensure all shared memory loads are complete
    __syncthreads();

    // Only compute if within image boundaries
    if (out_row < image_height && out_col < image_width)
    {
        float sum = 0.0f;

        // Unroll the convolution loops
        #pragma unroll
        for (int i = 0; i < KERNEL_HEIGHT; ++i)
        {
            #pragma unroll
            for (int j = 0; j < KERNEL_WIDTH; ++j)
            {
                sum += shared_mem[(threadIdx.y + i) * shared_width + (threadIdx.x + j)] * d_kernel_const[i * KERNEL_WIDTH + j];
            }
        }

        // Write the result to the output image
        d_output[out_row * image_width + out_col] = sum;
    }
}

// Function to launch the optimized convolution kernel
void launchConv2DOptimizedKernel(const float* d_input,
                                 float* d_output,
                                 int image_width,
                                 int image_height,
                                 int kernel_width,
                                 int kernel_height,
                                 dim3 grid,
                                 dim3 block,
                                 size_t shared_mem_size)
{
    conv2DOptimized<<<grid, block, shared_mem_size>>>(d_input, d_output,
                                                     image_width, image_height,
                                                     kernel_width, kernel_height);
}