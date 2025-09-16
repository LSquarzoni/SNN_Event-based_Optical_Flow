cmake_minimum_required(VERSION 3.10)
project(ONNX_LIF_operator)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set ONNX Runtime paths
set(ONNXRUNTIME_DIR "/scratch/msc25h1/onnxruntime")
include_directories(${ONNXRUNTIME_DIR}/include)
link_directories(${ONNXRUNTIME_DIR}/lib)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

# LibTorch C++ paths
set(Torch_DIR "/scratch/msc25h1/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)
include_directories("/scratch/msc25h1/libtorch/include")
include_directories("/scratch/msc25h1/libtorch/include/torch/csrc/api/include")

# Add the LIF operator library
add_library(lif_op SHARED src/lif_op.cpp)

# Link ONNX and LibTorch Runtime
target_link_libraries(lif_op onnxruntime)
target_link_libraries(lif_op "${TORCH_LIBRARIES}")

# Set the output directory for the shared library
set_target_properties(lif_op PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/python
)

# Install the library
install(TARGETS lif_op
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)