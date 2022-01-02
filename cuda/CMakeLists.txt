cmake_minimum_required(VERSION 3.9)

set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)

project(MyRayTracer_CUDA)

enable_language(CUDA)

set(CMAKE_CXX_STANDARD 17)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/thirdparty/include)

file(GLOB_RECURSE HEADERS ${PROJECT_SOURCE_DIR}/include/*.h)
source_group("Header Files" FILES ${HEADERS})
aux_source_directory(${PROJECT_SOURCE_DIR}/src/ DIR_SRCS)

################################

add_executable(${PROJECT_NAME} ${DIR_SRCS} ${HEADERS})
target_link_libraries(${PROJECT_NAME} glfw3)