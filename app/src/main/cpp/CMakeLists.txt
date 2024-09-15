# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html.
# For more examples on how to use CMake, see https://github.com/android/ndk-samples.

# Sets the minimum CMake version required for this project.
cmake_minimum_required(VERSION 3.22.1)


project(yolov8ncnn)


set(ncnn_DIR ${CMAKE_SOURCE_DIR}/ncnn-20240410-android-vulkan/${ANDROID_ABI}/lib/cmake/ncnn)
find_package(ncnn REQUIRED)


add_library(yolov8ncnn SHARED
        yolov8.cpp
        yolov8cls_Interface.cpp
        yolov8cls.cpp)

target_link_libraries(yolov8ncnn ncnn android log)