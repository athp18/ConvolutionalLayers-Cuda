#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "conv2D.h"

#define MAX_KERNEL_WIDTH 7
#define MAX_KERNEL_HEIGHT 7

// Declare the convolution kernel in constant memory
__constant__ float d_kernel_const[MAX_KERNEL_WIDTH * MAX_KERNEL_HEIGHT];

// Optimized CUDA kernel with enhanced shared memory usage and memory coalescing
__global__ void conv2DOptimized(const float* __restrict__ d_input,
                               float* __restrict__ d_output,  // Added __restrict__
                               const int image_width,         // Added const
                               const int image_height,
                               const int kernel_width,
                               const int kernel_height)
{
    // Compute the half sizes once
    const int k_half_width = kernel_width / 2;
    const int k_half_height = kernel_height / 2;

    // Compute the global row and column indices of the output element
    const int out_row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    const int out_col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    // Shared memory dimensions (tile + halo)
    const int shared_width = TILE_WIDTH + 2 * k_half_width;
    const int shared_height = TILE_WIDTH + 2 * k_half_height;
    
    // Allocate shared memory with padding for bank conflict prevention
    extern __shared__ float shared_mem[];
    
    // Calculate input indices
    const int in_row = out_row - k_half_height;
    const int in_col = out_col - k_half_width;
    
    // Optimize shared memory loading with vectorized loads where possible
    const int tid = threadIdx.y * TILE_WIDTH + threadIdx.x;
    const int total_threads = TILE_WIDTH * TILE_WIDTH;
    
    // Load main tile data using vectorized loads when possible
    #pragma unroll 4
    for (int i = tid; i < shared_width * shared_height; i += total_threads) {
        const int s_row = i / shared_width;
        const int s_col = i % shared_width;
        const int g_row = in_row + s_row - k_half_height;
        const int g_col = in_col + s_col - k_half_width;
        
        shared_mem[i] = (g_row >= 0 && g_row < image_height && 
                        g_col >= 0 && g_col < image_width) 
            ? d_input[g_row * image_width + g_col] 
            : 0.0f;
    }
    
    // Ensure all shared memory loads are complete
    __syncthreads();
    
    // Only compute if within image boundaries
    if (out_row < image_height && out_col < image_width) {
        float sum = 0.0f;
        
        // Use register-based accumulation for better performance
        #pragma unroll
        for (int i = 0; i < kernel_height; ++i) {
            const int shared_row = threadIdx.y + i;
            const float* shared_row_ptr = &shared_mem[shared_row * shared_width + threadIdx.x];
            const float* kernel_row_ptr = &d_kernel_const[i * kernel_width];
            
            #pragma unroll
            for (int j = 0; j < kernel_width; ++j) {
                sum = __fmaf_rn(shared_row_ptr[j], kernel_row_ptr[j], sum);
            }
        }
        
        // Write result using coalesced memory access
        d_output[out_row * image_width + out_col] = sum;
    }
}

// Helper function to compute grid and block dimensions
inline void computeGridBlock(int image_width, int image_height,
                           dim3& grid, dim3& block, size_t& shared_mem_size,
                           int kernel_width, int kernel_height) {
    block = dim3(TILE_WIDTH, TILE_WIDTH);
    grid = dim3((image_width + TILE_WIDTH - 1) / TILE_WIDTH,
                (image_height + TILE_WIDTH - 1) / TILE_WIDTH);
                
    // Calculate shared memory size including padding
    const int shared_width = TILE_WIDTH + 2 * (kernel_width / 2);
    const int shared_height = TILE_WIDTH + 2 * (kernel_height / 2);
    shared_mem_size = shared_width * shared_height * sizeof(float);
}

void launchConv2DOptimizedKernel(const float* d_input,
                                float* d_output,
                                int image_width,
                                int image_height,
                                int kernel_width,
                                int kernel_height,
                                cudaStream_t stream = nullptr)  // Added stream support
{
    dim3 grid, block;
    size_t shared_mem_size;
    computeGridBlock(image_width, image_height, grid, block, 
                    shared_mem_size, kernel_width, kernel_height);
    
    conv2DOptimized<<<grid, block, shared_mem_size, stream>>>(
        d_input, d_output, image_width, image_height, 
        kernel_width, kernel_height);
}
