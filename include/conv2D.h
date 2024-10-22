// conv2D.h
#ifndef CONV2D_H
#define CONV2D_H

#include <cuda_runtime.h>

// Launch function for basic convolution
void launchConv2DBasicKernel(const float* d_input,
                              const float* d_kernel,
                              float* d_output,
                              int image_width,
                              int image_height,
                              int kernel_width,
                              int kernel_height,
                              dim3 grid,
                              dim3 block,
                              size_t shared_mem_size);

// Launch function for optimized convolution
void launchConv2DOptimizedKernel(const float* d_input,
                                 float* d_output,
                                 int image_width,
                                 int image_height,
                                 int kernel_width,
                                 int kernel_height,
                                 dim3 grid,
                                 dim3 block,
                                 size_t shared_mem_size);

#endif // CONV2D_H