cmake_minimum_required(VERSION 3.20)
project(polyamg_ml LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(POLYAMG_WITH_ORT "Enable ONNX Runtime integration" OFF)
option(POLYAMG_WITH_PYBIND "Enable optional pybind11 interactive module" OFF)

if(NOT DEFINED ENV{PETSC_DIR})
  message(FATAL_ERROR "PETSC_DIR is not set. Export PETSC_DIR before configuring.")
endif()
if(NOT DEFINED ENV{PETSC_ARCH})
  message(FATAL_ERROR "PETSC_ARCH is not set. Export PETSC_ARCH before configuring.")
endif()

set(PETSC_DIR $ENV{PETSC_DIR})
set(PETSC_ARCH $ENV{PETSC_ARCH})
set(PETSC_LIBDIR "${PETSC_DIR}/${PETSC_ARCH}/lib")
set(PETSC_INCDIR "${PETSC_DIR}/include")
set(PETSC_ARCH_INCDIR "${PETSC_DIR}/${PETSC_ARCH}/include")

find_library(PETSC_LIB petsc HINTS "${PETSC_LIBDIR}" REQUIRED)
find_package(MPI REQUIRED)

set(PETSC_VARIABLES_FILE "${PETSC_LIBDIR}/petsc/conf/petscvariables")
if(EXISTS "${PETSC_VARIABLES_FILE}")
  file(READ "${PETSC_VARIABLES_FILE}" PETSC_VARS_TXT)
  string(REGEX MATCH "PETSC_CC_INCLUDES[^\n]*" PETSC_CC_INCLUDES_LINE "${PETSC_VARS_TXT}")
  if(PETSC_CC_INCLUDES_LINE)
    string(REGEX REPLACE "^[^=]*= *" "" PETSC_CC_INCLUDES "${PETSC_CC_INCLUDES_LINE}")
    separate_arguments(PETSC_CC_INCLUDES_LIST UNIX_COMMAND "${PETSC_CC_INCLUDES}")
  endif()
endif()

add_library(polyamg_core
  cpp_core/src/config.cpp
  cpp_core/src/schemas.cpp
  cpp_core/src/mesh_adapter_dmplex.cpp
  cpp_core/src/discretization_elliptic.cpp
  cpp_core/src/feature_extractor.cpp
  cpp_core/src/amg_solver.cpp
  cpp_core/src/amg_policy_grid.cpp
  cpp_core/src/amg_policy_factory.cpp
  cpp_core/src/experiment_runner.cpp
  cpp_core/src/experiment/result_sink.cpp
  cpp_core/src/bindings/interactive_api.cpp
)

target_include_directories(polyamg_core PUBLIC
  cpp_core/include
  "${PETSC_INCDIR}"
  "${PETSC_ARCH_INCDIR}"
  ${MPI_CXX_INCLUDE_DIRS}
  ${PETSC_CC_INCLUDES_LIST}
)

target_link_libraries(polyamg_core PUBLIC
  "${PETSC_LIB}"
  MPI::MPI_CXX
)

target_compile_definitions(polyamg_core PUBLIC PETSC_USE_LOG)

if(POLYAMG_WITH_ORT)
  if(DEFINED ENV{ONNXRUNTIME_ROOT})
    set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}")
  endif()
  find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
    HINTS
      "${ONNXRUNTIME_ROOT}/include"
      "${ONNXRUNTIME_ROOT}/include/onnxruntime"
      "${ONNXRUNTIME_ROOT}/include/onnxruntime/core/session"
      /usr/local/include
      /usr/local/include/onnxruntime
      /usr/local/include/onnxruntime/core/session
      /opt/homebrew/include
      /opt/homebrew/include/onnxruntime
      /opt/homebrew/include/onnxruntime/core/session
    PATH_SUFFIXES
      ""
      onnxruntime
      onnxruntime/core/session
  )
  if(NOT ONNXRUNTIME_INCLUDE_DIR)
    # Some distributions place headers under include/onnxruntime/core/session.
    find_path(ONNXRUNTIME_INCLUDE_DIR_ALT onnxruntime_cxx_api.h
      HINTS
        "${ONNXRUNTIME_ROOT}/include/onnxruntime"
        "${ONNXRUNTIME_ROOT}/include/onnxruntime/core/session"
        /usr/local/include/onnxruntime
        /usr/local/include/onnxruntime/core/session
        /opt/homebrew/include/onnxruntime
        /opt/homebrew/include/onnxruntime/core/session
    )
    if(ONNXRUNTIME_INCLUDE_DIR_ALT)
      set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_INCLUDE_DIR_ALT}")
    endif()
  endif()
  find_library(ONNXRUNTIME_LIB onnxruntime
    HINTS
      "${ONNXRUNTIME_ROOT}/lib"
      "${ONNXRUNTIME_ROOT}/lib64"
      /usr/local/lib
      /opt/homebrew/lib
  )

  if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIB)
    message(FATAL_ERROR
      "POLYAMG_WITH_ORT=ON but ONNX Runtime not found.\n"
      "ONNXRUNTIME_ROOT='${ONNXRUNTIME_ROOT}'\n"
      "ONNXRUNTIME_INCLUDE_DIR='${ONNXRUNTIME_INCLUDE_DIR}'\n"
      "ONNXRUNTIME_LIB='${ONNXRUNTIME_LIB}'")
  endif()

  message(STATUS "ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIR}")
  message(STATUS "ONNX Runtime lib: ${ONNXRUNTIME_LIB}")
  target_include_directories(polyamg_core PUBLIC "${ONNXRUNTIME_INCLUDE_DIR}")
  target_link_libraries(polyamg_core PUBLIC "${ONNXRUNTIME_LIB}")
  target_compile_definitions(polyamg_core PUBLIC POLYAMG_WITH_ORT)
endif()

add_executable(polyamg_baseline cpp_core/apps/main_baseline.cpp)
add_executable(polyamg_ann_amg cpp_core/apps/main_ann_amg.cpp)

target_link_libraries(polyamg_baseline PRIVATE polyamg_core)
target_link_libraries(polyamg_ann_amg PRIVATE polyamg_core)

if(POLYAMG_WITH_PYBIND)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(_polyamg_core cpp_core/src/bindings/pybind_module.cpp)
  target_link_libraries(_polyamg_core PRIVATE polyamg_core)
endif()

install(TARGETS polyamg_baseline polyamg_ann_amg RUNTIME DESTINATION bin)
