cmake_minimum_required(VERSION 3.10)
project(ColorSpaceConversion)

# 查找 OpenCV
find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV 未找到，请确保已正确安装。")
endif()

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "Libraries: ${OpenCV_LIBS}")
message(STATUS "Include dirs: ${OpenCV_INCLUDE_DIRS}")

# 添加可执行文件（注意路径：src/main.cpp）
add_executable(ColorSpaceConversion src/main.cpp)

# 链接 OpenCV 库
target_link_libraries(ColorSpaceConversion ${OpenCV_LIBS})

# 设置输出目录
set(OUTPUT_DIR ${CMAKE_SOURCE_DIR}/output)

# 在构建时创建输出目录
file(MAKE_DIRECTORY ${OUTPUT_DIR})

# 可选：将输出目录传递给编译器作为预处理器宏
add_definitions(-DOUTPUT_DIR="${OUTPUT_DIR}")
