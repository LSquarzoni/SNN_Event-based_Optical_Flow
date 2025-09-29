cmake_minimum_required(VERSION 3.10)
project(ONNX_LIF_operator)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include project src
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

# LibTorch C++ (use conda-installed torch)
# If Torch_DIR isn't provided by the environment, default to the path inside the active conda env
if(NOT DEFINED Torch_DIR AND DEFINED ENV{CONDA_PREFIX})
    set(Torch_DIR "$ENV{CONDA_PREFIX}/lib/python3.9/site-packages/torch/share/cmake/Torch")
endif()
find_package(Torch REQUIRED)
include_directories("${Torch_DIR}/../../../include")
include_directories("${Torch_DIR}/../../../include/torch/csrc/api/include")

# Add the LIF operator library
add_library(lif_op SHARED src/lif_op.cpp)

# Link LibTorch Runtime only (no ONNX Runtime required for this op)
target_link_libraries(lif_op "${TORCH_LIBRARIES}")

# Set the output directory for the shared library
set_target_properties(lif_op PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/python
)

# Install the library
install(TARGETS lif_op
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)