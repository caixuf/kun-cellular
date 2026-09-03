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
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(kun_cellular INTERFACE Threads::Threads m)

enable_testing()

# 1. 编译运行 C++ 宏观生态与形态发生单元测试
file(GLOB TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_flow_*.cpp")
foreach(TEST_SRC ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SRC} NAME_WE)
    add_executable(${TEST_NAME} ${TEST_SRC})
    target_link_libraries(${TEST_NAME} PRIVATE kun_cellular)
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
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


