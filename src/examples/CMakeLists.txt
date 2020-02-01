find_package(CUDA)
find_package(Threads)

vkt_use_package(CUDA)
vkt_use_package(Threads)

include_directories(${PROJECT_SOURCE_DIR}/include/c)
include_directories(${PROJECT_SOURCE_DIR}/include/cpp)

vkt_link_libraries(volkit)

vkt_add_executable(CoreAlgorithmsC CoreAlgorithms.c)
vkt_add_executable(DecomposeC Decompose.c)
vkt_add_executable(ExecutionPolicyC ExecutionPolicy.c)
vkt_add_executable(ScanC Scan.c)

vkt_add_executable(DecomposeCPP Decompose.cpp)
vkt_add_executable(ExecutionPolicyCPP ExecutionPolicy.cpp)
