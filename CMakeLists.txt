# CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(cuda-convolution LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Enable testing
enable_testing()

# Find CUDA
find_package(CUDA REQUIRED)

# Include directories
include_directories(include)

# Source files
set(SOURCES
    src/main.cu
    src/conv2D.cu
    src/utils.cu
)

# Add executable
add_executable(conv2D ${SOURCES})

# Specify CUDA properties
set_target_properties(conv2D PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "60;61;70;75;80;86" # Adjust based on your GPU
)

# Link libraries if necessary
# target_link_libraries(conv2D PRIVATE ...)

# Add tests
add_subdirectory(tests)