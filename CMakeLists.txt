set(TRITON_X86_POC_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(TRITON_X86_POC_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include) # Tablegen'd files 

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../triton_shared/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../triton_shared/include)

add_subdirectory(test)
add_subdirectory(tools)
