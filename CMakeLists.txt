cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(SimpleConcurrentGPUHashTable LANGUAGES CXX CUDA)

# put predefined cmake projects in their own solution folder
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(CMAKE_BUILD_TYPE Debug)

# string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_70,code=sm_70")

set(SOURCE_FILES src/main.cpp src/test.cpp src/linearprobing.h src/linearprobing.cu)

#set(SOURCE_FILES src/kernel_1MB.cu)

include_directories(${CMAKE_SOURCE_DIR}/src)

add_executable(test ${SOURCE_FILES})
#add_executable(kernel_1MB ${SOURCE_FILES})
set_target_properties(test PROPERTIES CUDA_ARCHITECTURES "70")

# visual studio project should mimic directory structure
# this isn't working for me; I think because
# https://developercommunity.visualstudio.com/content/problem/777578/source-grouptree-no-longer-works.html
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCE_FILES})
