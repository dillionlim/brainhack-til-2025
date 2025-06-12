cmake_minimum_required(VERSION 3.14)
project(surprise_server)

# 1) Force Release with -O3, arch tuning, unrolling, and LTO
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Global C++ flags for Release
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -funroll-loops -flto -fomit-frame-pointer")

# 4) Enable CMake’s Interprocedural Optimization (LTO)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)

find_package(OpenCV REQUIRED)
find_package(Crow CONFIG REQUIRED)
find_package(ZLIB REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(OpenMP)  # 5) OpenMP for parallel SSIM loops

add_executable(surprise
    server.cpp
    metrics.cpp
    SurpriseManager.cpp
    base64.cpp
)

# 5) Link and compile with OpenMP if available
if(OpenMP_CXX_FOUND)
  target_link_libraries(surprise PRIVATE OpenMP::OpenMP_CXX)
endif()

target_link_libraries(surprise PRIVATE
    Crow::Crow
    ${OpenCV_LIBS}
    ZLIB::ZLIB
    pthread
    OpenSSL::Crypto
    OpenSSL::SSL
)
