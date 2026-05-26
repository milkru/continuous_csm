cmake_minimum_required(VERSION 3.22)
project(continuous_csm)

set(CMAKE_CXX_STANDARD 20)

find_package(Python3 REQUIRED)

execute_process(
    COMMAND ${Python3_EXECUTABLE} deploy_content.py
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/3rdparty/lightweightvk
)

execute_process(
    COMMAND ${Python3_EXECUTABLE} deploy_deps.py
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/3rdparty/lightweightvk
)

set(LVK_WITH_SAMPLES OFF CACHE BOOL "" FORCE)

add_subdirectory(3rdparty/lightweightvk)

add_executable(continuous_csm src/main.cpp)

target_link_libraries(continuous_csm PRIVATE LVKLibrary)

set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT continuous_csm)
