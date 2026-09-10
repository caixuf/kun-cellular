cmake_minimum_required(VERSION 3.16)
project(KunCellular VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC -Wall -Wextra")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -fPIC -Wall -Wextra")
string(REPLACE "-DNDEBUG" "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
string(REPLACE "-DNDEBUG" "" CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}")
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fcoroutines")
endif()

find_package(Threads REQUIRED)
option(KUN_BUILD_KNOWLEDGE "Build peripheral R8 native knowledge library and experiment" ON)
if(KUN_BUILD_KNOWLEDGE)
    find_package(SQLite3 REQUIRED)
    find_package(OpenSSL REQUIRED COMPONENTS Crypto)
    add_library(kun_knowledge INTERFACE)
    target_link_libraries(kun_knowledge INTERFACE SQLite::SQLite3 OpenSSL::Crypto)
endif()
include(CheckCXXCompilerFlag)

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

# L1 系统层 INTERFACE 库 (header-only; 仅消费 L0 公开 API, 领域无关)
add_library(kun_population INTERFACE)
target_link_libraries(kun_population INTERFACE kun_cellular)

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
    if(TEST_NAME STREQUAL "test_germline_library_cpp" AND NOT KUN_BUILD_KNOWLEDGE)
        continue()
    endif()
    if(TEST_NAME STREQUAL "test_assert_canary")
        add_executable(${TEST_NAME} ${TEST_SRC})
        target_link_libraries(${TEST_NAME} PRIVATE kun_cellular)
        target_compile_options(${TEST_NAME} PRIVATE -UNDEBUG)
        # 断言金丝雀门禁：验证断言处于活体状态（assert(false) 必定 abort，取反为 0）；若断言被剥除则测试红
        add_test(NAME ${TEST_NAME} COMMAND bash -c "! $<TARGET_FILE:${TEST_NAME}>")
    elseif(TEST_NAME MATCHES "cuda_scale" OR TEST_NAME MATCHES "adas_scales")
        if(CUDA_HEADER_PATH AND CUDA_DRIVER_LIB AND NVRTC_LIB)
            add_executable(${TEST_NAME} ${TEST_SRC})
            target_include_directories(${TEST_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CUDA_HEADER_PATH})
            target_link_libraries(${TEST_NAME} PRIVATE kun_cellular ${CUDA_DRIVER_LIB} ${NVRTC_LIB})
            target_compile_options(${TEST_NAME} PRIVATE -UNDEBUG)
            add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
            # CUDA 重载测试并行偶发闪断 (单跑满分): 串行隔离
            set_tests_properties(${TEST_NAME} PROPERTIES RUN_SERIAL TRUE)
            if(EXISTS "$ENV{HOME}/.local/lib/python3.12/site-packages/nvidia/cu13/lib")
                set_tests_properties(${TEST_NAME} PROPERTIES ENVIRONMENT "LD_LIBRARY_PATH=$ENV{HOME}/.local/lib/python3.12/site-packages/nvidia/cu13/lib:$ENV{LD_LIBRARY_PATH}")
            endif()
        endif()
    else()
        add_executable(${TEST_NAME} ${TEST_SRC})
        if(TEST_NAME MATCHES "^test_population_")
            target_link_libraries(${TEST_NAME} PRIVATE kun_population)
        else()
            target_link_libraries(${TEST_NAME} PRIVATE kun_cellular)
        endif()
        target_compile_options(${TEST_NAME} PRIVATE -UNDEBUG)
        add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
        # 并行负载下时序/内存 flaky (单跑/串行满分): flow 家族重载实验, 统一串行隔离
        if(TEST_NAME MATCHES "^test_flow_")
            set_tests_properties(${TEST_NAME} PROPERTIES RUN_SERIAL TRUE)
        endif()
    endif()
endforeach()

if(KUN_BUILD_KNOWLEDGE)
    if(TARGET test_the_chain)
        target_link_libraries(test_the_chain PRIVATE kun_knowledge)
    endif()
    if(TARGET test_germline_library_cpp)
        target_link_libraries(test_germline_library_cpp PRIVATE kun_knowledge)
    endif()
    add_executable(knowledge_transfer_demo tasks/transfer/knowledge_transfer_demo.cpp)
    target_link_libraries(knowledge_transfer_demo PRIVATE kun_cellular kun_knowledge)
endif()

if(TARGET test_cellular_core_kernel)
    target_sources(test_cellular_core_kernel PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_cellular_core_kernel_c_bridge.c")
endif()
if(TARGET test_cellular_core_kernel_header)
    target_compile_options(test_cellular_core_kernel_header PRIVATE -pedantic-errors)
endif()
if(TARGET test_cellular_core_strict_kernel)
    target_sources(test_cellular_core_strict_kernel PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_cellular_core_kernel_c_bridge.c")
    target_compile_options(test_cellular_core_strict_kernel PRIVATE -pedantic-errors)
endif()
check_cxx_compiler_flag("-mlong-double-64" KUN_HAS_MLONG_DOUBLE_64)
if(TARGET test_cellular_core_bridge AND KUN_HAS_MLONG_DOUBLE_64)
    target_compile_options(test_cellular_core_bridge PRIVATE -mlong-double-64)
endif()

# 2. 编译运行纯 C11 硬件级核心原语、通用运行时与连续流体介质测试 (Single Source of Truth)
set(C_TEST_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_core_primitives.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_universal_runtime.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_multiphase_fluid_stress.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_binary_runtime_scale.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_cellular_core_kernel_c.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_cellular_core_strict_kernel_c.c"
)
foreach(C_TEST_SRC ${C_TEST_SOURCES})
    if(EXISTS ${C_TEST_SRC})
        get_filename_component(TEST_NAME ${C_TEST_SRC} NAME_WE)
        add_executable(${TEST_NAME} ${C_TEST_SRC})
        target_link_libraries(${TEST_NAME} PRIVATE kun_cellular m)
        target_compile_options(${TEST_NAME} PRIVATE -UNDEBUG)
        if(TEST_NAME STREQUAL "test_cellular_core_kernel_c" OR
           TEST_NAME STREQUAL "test_cellular_core_strict_kernel_c")
            target_compile_options(${TEST_NAME} PRIVATE -pedantic-errors)
        endif()
        add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
    endif()
endforeach()

# 3. 编译纯原生生境形态发生演化训练器 (Native Evolution Trainers)
add_executable(train_maze_navigator "${CMAKE_CURRENT_SOURCE_DIR}/tools/train_maze_navigator.cpp")
target_link_libraries(train_maze_navigator PRIVATE kun_cellular Threads::Threads m)

add_executable(train_maze_tripartite "${CMAKE_CURRENT_SOURCE_DIR}/tools/train_maze_tripartite.cpp")
target_link_libraries(train_maze_tripartite PRIVATE kun_cellular Threads::Threads m)

add_executable(train_household_coverage_tripartite "${CMAKE_CURRENT_SOURCE_DIR}/tools/train_household_coverage_tripartite.cpp")
target_link_libraries(train_household_coverage_tripartite PRIVATE kun_cellular Threads::Threads m)

add_executable(bench_doudizhu_moe "${CMAKE_CURRENT_SOURCE_DIR}/tools/bench_doudizhu_moe.cpp")
target_link_libraries(bench_doudizhu_moe PRIVATE kun_cellular Threads::Threads m)

add_executable(train_doudizhu_bc_distill "${CMAKE_CURRENT_SOURCE_DIR}/tools/train_doudizhu_bc_distill.cpp")
target_link_libraries(train_doudizhu_bc_distill PRIVATE kun_cellular Threads::Threads m)

  add_executable(p9_runner "${CMAKE_CURRENT_SOURCE_DIR}/tools/p9_runner.cpp")
  target_link_libraries(p9_runner PRIVATE kun_cellular Threads::Threads m)

  add_executable(p9_disagree "${CMAKE_CURRENT_SOURCE_DIR}/tools/p9_disagree.cpp")
  target_link_libraries(p9_disagree PRIVATE kun_cellular Threads::Threads m)

  add_executable(p9_online "${CMAKE_CURRENT_SOURCE_DIR}/tools/p9_online.cpp")
  target_link_libraries(p9_online PRIVATE kun_cellular Threads::Threads m)

  add_executable(p9_online2 "${CMAKE_CURRENT_SOURCE_DIR}/tools/p9_online2.cpp")
  target_link_libraries(p9_online2 PRIVATE kun_cellular Threads::Threads m)

  add_executable(p9_paired "${CMAKE_CURRENT_SOURCE_DIR}/tools/p9_paired.cpp")
  target_link_libraries(p9_paired PRIVATE kun_cellular Threads::Threads m)

  add_executable(p9_online3 "${CMAKE_CURRENT_SOURCE_DIR}/tools/p9_online3.cpp")
  target_link_libraries(p9_online3 PRIVATE kun_cellular Threads::Threads m)

  add_executable(bench_p9_speed "${CMAKE_CURRENT_SOURCE_DIR}/tools/bench_p9_speed.cpp")
  target_link_libraries(bench_p9_speed PRIVATE kun_cellular Threads::Threads m)

  add_executable(p9_fuse "${CMAKE_CURRENT_SOURCE_DIR}/tools/p9_fuse.cpp")
  target_link_libraries(p9_fuse PRIVATE kun_cellular Threads::Threads m)

  add_executable(p9_grow "${CMAKE_CURRENT_SOURCE_DIR}/tools/p9_grow.cpp")
  target_link_libraries(p9_grow PRIVATE kun_cellular Threads::Threads m)

add_executable(u5_growth_winner "${CMAKE_CURRENT_SOURCE_DIR}/tools/u5_growth_winner.cpp")
target_link_libraries(u5_growth_winner PRIVATE kun_cellular Threads::Threads m)

add_executable(u4_ablation "${CMAKE_CURRENT_SOURCE_DIR}/tools/u4_ablation.cpp")
target_link_libraries(u4_ablation PRIVATE kun_cellular Threads::Threads m)

add_executable(timing_discovery "${CMAKE_CURRENT_SOURCE_DIR}/tools/timing_discovery.cpp")
target_link_libraries(timing_discovery PRIVATE kun_cellular Threads::Threads m)

add_executable(u4_link6 "${CMAKE_CURRENT_SOURCE_DIR}/tools/u4_link6.cpp")
target_link_libraries(u4_link6 PRIVATE kun_cellular kun_knowledge Threads::Threads m)

add_executable(u4_task_level "${CMAKE_CURRENT_SOURCE_DIR}/tools/u4_task_level.cpp")
target_link_libraries(u4_task_level PRIVATE kun_cellular Threads::Threads m)

add_executable(train_quant_futures "${CMAKE_CURRENT_SOURCE_DIR}/tools/train_quant_futures.cpp")
target_link_libraries(train_quant_futures PRIVATE kun_cellular Threads::Threads m)

add_executable(train_multi_asset_quant_master "${CMAKE_CURRENT_SOURCE_DIR}/tools/train_multi_asset_quant_master.cpp")
target_link_libraries(train_multi_asset_quant_master PRIVATE kun_cellular Threads::Threads m)

add_executable(train_multi_asset_cortical_array "${CMAKE_CURRENT_SOURCE_DIR}/tools/train_multi_asset_cortical_array.cpp")
target_link_libraries(train_multi_asset_cortical_array PRIVATE kun_cellular Threads::Threads m)
target_include_directories(train_multi_asset_cortical_array PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tools")

add_executable(train_quant_population_ecology "${CMAKE_CURRENT_SOURCE_DIR}/tools/train_quant_population_ecology.cpp")
target_link_libraries(train_quant_population_ecology PRIVATE kun_population Threads::Threads m)
target_include_directories(train_quant_population_ecology PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tools")

# L1 演化速度基准 (报告工具, 非 ctest 门禁)
add_executable(bench_population_evolution "${CMAKE_CURRENT_SOURCE_DIR}/tools/bench_population_evolution.cpp")
target_link_libraries(bench_population_evolution PRIVATE kun_population Threads::Threads m)

add_executable(train_multi_asset_quant_l3 "${CMAKE_CURRENT_SOURCE_DIR}/tools/train_multi_asset_quant_l3.cpp")
target_link_libraries(train_multi_asset_quant_l3 PRIVATE kun_cellular Threads::Threads m)

add_executable(train_flagship_voxel "${CMAKE_CURRENT_SOURCE_DIR}/tools/train_flagship_voxel.cpp")
target_link_libraries(train_flagship_voxel PRIVATE kun_cellular Threads::Threads m)

add_executable(train_field_3d_turbulence "${CMAKE_CURRENT_SOURCE_DIR}/tools/train_field_3d_turbulence.cpp")
target_link_libraries(train_field_3d_turbulence PRIVATE kun_cellular Threads::Threads m)

add_executable(train_domain_zoo "${CMAKE_CURRENT_SOURCE_DIR}/tools/train_domain_zoo.cpp")
target_link_libraries(train_domain_zoo PRIVATE kun_cellular Threads::Threads m)

add_executable(kun_certify "${CMAKE_CURRENT_SOURCE_DIR}/tools/kun_certify.cpp")
target_link_libraries(kun_certify PRIVATE kun_cellular Threads::Threads m)

add_executable(train_cart_pole_tripartite "${CMAKE_CURRENT_SOURCE_DIR}/tools/train_cart_pole_tripartite.cpp")
target_link_libraries(train_cart_pole_tripartite PRIVATE kun_cellular Threads::Threads m)

add_executable(train_quant_tripartite "${CMAKE_CURRENT_SOURCE_DIR}/tools/train_quant_tripartite.cpp")
target_link_libraries(train_quant_tripartite PRIVATE kun_cellular Threads::Threads m)

add_executable(train_adas_tripartite "${CMAKE_CURRENT_SOURCE_DIR}/tools/train_adas_tripartite.cpp")
target_link_libraries(train_adas_tripartite PRIVATE kun_cellular Threads::Threads m)

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

add_library(kun_doudizhu_runtime SHARED
    "${CMAKE_CURRENT_SOURCE_DIR}/tools/doudizhu_c_runtime.cpp"
)
target_include_directories(kun_doudizhu_runtime PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(kun_doudizhu_runtime PRIVATE kun_cellular Threads::Threads m)
set_target_properties(kun_doudizhu_runtime PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

# 5. GPU vs C++ 硬件底座逐细胞全量数值保真对账测试与三方原语单一真相源门禁
find_package(Python3 COMPONENTS Interpreter)
if(Python3_FOUND)
    add_test(NAME test_gpu_cpp_bit_parity COMMAND ${Python3_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_gpu_cpp_bit_parity.py)
    set_tests_properties(test_gpu_cpp_bit_parity PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}:$ENV{PYTHONPATH}")

    add_test(NAME test_op_parity_codegen COMMAND ${Python3_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_op_parity_codegen.py)
    set_tests_properties(test_op_parity_codegen PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}:$ENV{PYTHONPATH}")

    add_test(NAME test_op_vjp_finite_difference COMMAND ${Python3_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_op_vjp_finite_difference.py)
    set_tests_properties(test_op_vjp_finite_difference PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}:$ENV{PYTHONPATH}")

    add_test(NAME test_formal_certification COMMAND ${CMAKE_CURRENT_BINARY_DIR}/kun_certify --organism ${CMAKE_CURRENT_SOURCE_DIR}/checkpoints/cartpole_balance_champion.bin --regression-steps 50000 --out ${CMAKE_CURRENT_BINARY_DIR}/cartpole.cert.json)
endif()
