
cmake_minimum_required(VERSION 3.10)
project(QuickTutorial_plugin_project)

if(NOT "${CMAKE_CXX_STANDARD}")
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_POSITION_INDEPENDENT_CODE ON)   # Enable -fPIC for all targets

# Expose custom cmake modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

# Set install directory to $JANA_HOME
set(CMAKE_INSTALL_PREFIX $ENV{JANA_HOME} CACHE PATH "magic incantation" FORCE)

add_subdirectory(src)
# add_subdirectory(tests)

