cmake_minimum_required(VERSION 3.23)

include(../CMake/CommonMacros.txt)

set(PROJECT_NAME "Core")
project(${PROJECT_NAME} CXX CUDA)

file(GLOB_RECURSE SRC_FILES LIST_DIRECTORIES false RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} src/*.c?? *.cu)
file(GLOB_RECURSE HEADER_FILES LIST_DIRECTORIES false RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} src/*.h)

include_directories(src)

link_directories(${CMAKE_SOURCE_DIR}/deps/lib)

add_executable(${PROJECT_NAME} ${SRC_FILES} ${HEADER_FILES})

# you would also need to put the dll along side with exe
target_link_libraries(${PROJECT_NAME} glut32 glut64)

if(MSVC)
    # for convenience: we don't need Microsoft C++ runtime to run the project
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
    # set current working directoy so that relative path works
    set_property(TARGET ${PROJECT_NAME} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src/")
endif()
 
SETUP_GROUPS("${SRC_FILES}")
SETUP_GROUPS("${HEADER_FILES}")

SET_OUTPUT_NAMES(${PROJECT_NAME})
# CUDA supports up to C++14
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 14)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)


