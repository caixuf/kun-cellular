cmake_minimum_required(VERSION 3.16)
project(KunCellular VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG -fPIC -Wall -Wextra")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -DNDEBUG -fPIC -Wall -Wextra")
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fcoroutines")
endif()

find_package(Threads REQUIRED)

add_library(kun_cellular INTERFACE)
target_include_directories(kun_cellular INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(kun_cellular INTERFACE Threads::Threads m)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(kun_cellular INTERFACE OpenMP::OpenMP_CXX)
endif()

enable_testing()

# 自动探测 CUDA Driver 与 NVRTC 支持
find_path(CUDA_HEADER_PATH cuda.h
    PATHS
    /usr/local/cuda/include
    /usr/include
    $ENV{HOME}/.local/lib/python3.12/site-packages/nvidia/cu13/include
)
find_library(CUDA_DRIVER_LIB cuda
    PATHS
    /usr/lib/wsl/lib
    /usr/lib/x86_64-linux-gnu
    /usr/local/cuda/lib64
)
find_library(NVRTC_LIB nvrtc
    PATHS
    $ENV{HOME}/.local/lib/python3.12/site-packages/nvidia/cu13/lib
    /usr/local/cuda/lib64
    /usr/lib/x86_64-linux-gnu
)

# 1. 编译运行 C++ 宏观生态与形态发生单元测试
file(GLOB TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_*.cpp")
foreach(TEST_SRC ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SRC} NAME_WE)
    if(TEST_NAME MATCHES "cuda_scale" OR TEST_NAME MATCHES "adas_scales")
        if(CUDA_HEADER_PATH AND CUDA_DRIVER_LIB AND NVRTC_LIB)
            add_executable(${TEST_NAME} ${TEST_SRC})
            target_include_directories(${TEST_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CUDA_HEADER_PATH})
            target_link_libraries(${TEST_NAME} PRIVATE kun_cellular ${CUDA_DRIVER_LIB} ${NVRTC_LIB})
            add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
            if(EXISTS "$ENV{HOME}/.local/lib/python3.12/site-packages/nvidia/cu13/lib")
                set_tests_properties(${TEST_NAME} PROPERTIES ENVIRONMENT "LD_LIBRARY_PATH=$ENV{HOME}/.local/lib/python3.12/site-packages/nvidia/cu13/lib:$ENV{LD_LIBRARY_PATH}")
            endif()
        endif()
    else()
        add_executable(${TEST_NAME} ${TEST_SRC})
        target_link_libraries(${TEST_NAME} PRIVATE kun_cellular)
        add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
    endif()
endforeach()

# 2. 编译运行纯 C11 硬件级核心原语、通用运行时与连续流体介质测试 (Single Source of Truth)
set(C_TEST_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_core_primitives.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_universal_runtime.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_multiphase_fluid_stress.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_binary_runtime_scale.c"
)
foreach(C_TEST_SRC ${C_TEST_SOURCES})
    if(EXISTS ${C_TEST_SRC})
        get_filename_component(TEST_NAME ${C_TEST_SRC} NAME_WE)
        add_executable(${TEST_NAME} ${C_TEST_SRC})
        target_link_libraries(${TEST_NAME} PRIVATE kun_cellular m)
        add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
    endif()
endforeach()

# 3. 编译纯原生生境形态发生演化训练器 (Native Evolution Trainers)
add_executable(train_maze_navigator "${CMAKE_CURRENT_SOURCE_DIR}/tools/train_maze_navigator.cpp")
target_link_libraries(train_maze_navigator PRIVATE kun_cellular Threads::Threads m)

add_executable(train_doudizhu_champion "${CMAKE_CURRENT_SOURCE_DIR}/tools/train_doudizhu_champion.cpp")
target_link_libraries(train_doudizhu_champion PRIVATE kun_cellular Threads::Threads m)

add_executable(train_quant_futures "${CMAKE_CURRENT_SOURCE_DIR}/tools/train_quant_futures.cpp")
target_link_libraries(train_quant_futures PRIVATE kun_cellular Threads::Threads m)

add_executable(train_multi_asset_quant_master "${CMAKE_CURRENT_SOURCE_DIR}/tools/train_multi_asset_quant_master.cpp")
target_link_libraries(train_multi_asset_quant_master PRIVATE kun_cellular Threads::Threads m)

add_executable(train_multi_asset_cortical_array "${CMAKE_CURRENT_SOURCE_DIR}/tools/train_multi_asset_cortical_array.cpp")
target_link_libraries(train_multi_asset_cortical_array PRIVATE kun_cellular Threads::Threads m)

add_executable(train_multi_asset_quant_l3 "${CMAKE_CURRENT_SOURCE_DIR}/tools/train_multi_asset_quant_l3.cpp")
target_link_libraries(train_multi_asset_quant_l3 PRIVATE kun_cellular Threads::Threads m)

add_executable(train_flagship_voxel "${CMAKE_CURRENT_SOURCE_DIR}/tools/train_flagship_voxel.cpp")
target_link_libraries(train_flagship_voxel PRIVATE kun_cellular Threads::Threads m)

add_executable(train_field_3d_turbulence "${CMAKE_CURRENT_SOURCE_DIR}/tools/train_field_3d_turbulence.cpp")
target_link_libraries(train_field_3d_turbulence PRIVATE kun_cellular Threads::Threads m)

add_executable(train_domain_zoo "${CMAKE_CURRENT_SOURCE_DIR}/tools/train_domain_zoo.cpp")
target_link_libraries(train_domain_zoo PRIVATE kun_cellular Threads::Threads m)

if(CUDA_HEADER_PATH AND CUDA_DRIVER_LIB AND NVRTC_LIB)
    add_executable(bench_adas_scales "${CMAKE_CURRENT_SOURCE_DIR}/tools/bench_adas_scales.cpp")
    target_include_directories(bench_adas_scales PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CUDA_HEADER_PATH})
    target_link_libraries(bench_adas_scales PRIVATE kun_cellular ${CUDA_DRIVER_LIB} ${NVRTC_LIB})
endif()

# 4. 构建纯 C11 硬件级通用运行时动态共享库 (C-ABI Shared Library for ctypes / Python Binding)
add_library(kun_cellular_runtime SHARED
    "${CMAKE_CURRENT_SOURCE_DIR}/src/sdsc_c_runtime.c"
)
target_include_directories(kun_cellular_runtime PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(kun_cellular_runtime PRIVATE m)
set_target_properties(kun_cellular_runtime PROPERTIES
    C_STANDARD 11
    C_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

# 5. GPU vs C++ 硬件底座逐细胞全量数值保真对账测试与三方原语单一真相源门禁
find_package(Python3 COMPONENTS Interpreter)
if(Python3_FOUND)
    add_test(NAME test_gpu_cpp_bit_parity COMMAND ${Python3_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_gpu_cpp_bit_parity.py)
    set_tests_properties(test_gpu_cpp_bit_parity PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}:$ENV{PYTHONPATH}")

    add_test(NAME test_op_parity_codegen COMMAND ${Python3_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_op_parity_codegen.py)
    set_tests_properties(test_op_parity_codegen PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}:$ENV{PYTHONPATH}")
endif()
