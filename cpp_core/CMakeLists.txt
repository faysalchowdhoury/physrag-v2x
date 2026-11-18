cmake_minimum_required(VERSION 3.15)
project(physrag_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(physrag_cpp bindings.cpp)
target_include_directories(physrag_cpp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Statically link the MinGW C++/GCC runtime so the module has no external DLL
# dependency (Python 3.8+ does not search PATH for extension dependencies).
if(MINGW)
  target_link_options(physrag_cpp PRIVATE
    -static -static-libgcc -static-libstdc++)
endif()
