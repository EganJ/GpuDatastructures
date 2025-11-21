cmake_minimum_required(VERSION 3.18)

project(gpudatastructures LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE Debug)
# set(CMAKE_CUDA_FLAGS_DEBUG "-G")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -rdc=true")

# Collect all source files
file(GLOB_RECURSE KERNEL_SOURCES CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/*.cu")
file(GLOB_RECURSE LIB_SOURCES CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
list(REMOVE_ITEM LIB_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")

add_executable(main src/main.cpp ${KERNEL_SOURCES} ${LIB_SOURCES})
# target_link_libraries(main PRIVATE gpudslib)
set_target_properties(main PROPERTIES
     CUDA_ARCHITECTURES "89"
     CUDA_SEPARABLE_COMPILATION ON
)
