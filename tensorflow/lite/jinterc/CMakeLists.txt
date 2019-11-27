cmake_minimum_required(VERSION 3.2)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

set(project_name "jinterc")
set(project_version "0.1")

project("${project_name}")

if(NOT DEFINED ENV{SNPS_AI_INSTALL_DIR})
    message( FATAL_ERROR "Environment variable SNPS_AI_INSTALL_DIR is required")
endif()

set(SNPS_AI_INSTALL_DIR $ENV{SNPS_AI_INSTALL_DIR})
set(CMAKE_INSTALL_PREFIX "${SNPS_AI_INSTALL_DIR}")

set(LOCAL_CFLAGS " -DGEMMLOWP_ALLOW_SLOW_SCALAR_FALLBACK -pthread")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${LOCAL_CFLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${LOCAL_CFLAGS}")

file(GLOB INSTALL_INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h" )

file(GLOB LIB_SRC_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc" 
)

file(GLOB MIN_SRC_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/mintest/*.cc" 
)

include_directories(BEFORE
    ${CMAKE_CURRENT_BINARY_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../..
)

message(STATUS "About to include TfLiteDecls...")
include(TfLiteDecls)
# message(STATUS "TFLITE_SRC: ${TFLITE_SRC}")
message(STATUS "After include of TfLiteDecls...")

list(APPEND LIB_SRC_FILES ${TFLITE_SRC})

# add_executable(mintest ${LIB_SRC_FILES} ${MIN_SRC_FILES})
add_executable(mintest ${MIN_SRC_FILES})
target_link_libraries(mintest ${project_name})

# -- The TFL makefile is a mess, it includes lots of src files that define
# main.  Therefore, can't build shared lib....
#
#add_library(${project_name} SHARED ${LIB_SRC_FILES})

add_library(${project_name} ${LIB_SRC_FILES})

# -- The rest of this file is boilerplate installation & packaging
#
target_include_directories (${project_name} 
    PUBLIC 
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/internal>
)

install (TARGETS ${project_name}
    EXPORT ${project_name}Targets
    DESTINATION lib
)

add_library(${project_name}::${project_name} ALIAS ${project_name})

install (EXPORT ${project_name}Targets
    FILE
        ${project_name}Targets.cmake
    NAMESPACE
        ${project_name}::
    DESTINATION
        lib/cmake/${project_name}
)

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/${project_name}ConfigVersion.cmake
    VERSION ${project_version}
    COMPATIBILITY AnyNewerVersion
)

install (FILES
    ${CMAKE_CURRENT_LIST_DIR}/cmake/${project_name}Config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/${project_name}ConfigVersion.cmake
    DESTINATION
        lib/cmake/${project_name}
)

install(FILES ${INSTALL_INCLUDES} DESTINATION include/${project_name})

