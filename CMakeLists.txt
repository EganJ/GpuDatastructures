cmake_minimum_required(VERSION 3.18)

project(gpudatastructures LANGUAGES CXX CUDA)

# Collect all source files
file(GLOB_RECURSE KERNEL_SOURCES CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/*.cu")
file(GLOB_RECURSE LIB_SOURCES CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
list(REMOVE_ITEM LIB_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")

add_library(gpudslib ${KERNEL_SOURCES} ${LIB_SOURCES})
target_include_directories(gpudslib PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")

set_target_properties(gpudslib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "60;70;75;80;86"
)

add_executable(main src/main.cpp)
target_link_libraries(main PRIVATE gpudslib)
set_target_properties(main PROPERTIES
     CUDA_ARCHITECTURES "60;70;75;80;86"
)

add_executable(unionfind_test_generator src/make_unionfind_test.cpp)