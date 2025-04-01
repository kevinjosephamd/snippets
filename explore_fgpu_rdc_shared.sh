#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

mkdir -p ${SCRIPT_DIR}/build
rm ${SCRIPT_DIR}/build/*
cd ${SCRIPT_DIR}/build

cat > library_interface.h << EOF
#pragma once
using kernel_type = __global__ void (*)(int*);
auto get_kernel() -> kernel_type;
EOF

cat > library1.hip << EOF
#include<hip/hip_runtime.h>
#include "library_interface.h"

__global__ void my_kernel(int*);
using kernel_type = decltype(&my_kernel);

__device__ auto add_one(int value) -> int {
  return value + 1;
}

__device__ auto get_device_function() {
  return add_one;
}

auto get_kernel() -> kernel_type {
  return my_kernel;
}
EOF

cat > library2.hip << EOF
#include<hip/hip_runtime.h>

using device_function_type = int(*)(int value);

__device__ device_function_type get_device_function();

__global__ void my_kernel(int* inout) {
  printf("Hello from my_kernel\n");
  int value = 1;
  value = get_device_function()(value);
  *inout = value;
}
EOF

cat > main.hip << EOF
#include<hip/hip_runtime.h>
#include <stdio.h>
#include "library_interface.h"

#define HIP_CHECK(call)                                                        \
  do {                                                                         \
    hipError_t err = call;                                                     \
    if (err != hipSuccess) {                                                   \
      throw std::runtime_error(std::string("HIP error at ") + __FILE__ + ":" + \
                               std::to_string(__LINE__) +                      \
                               ", code=" + std::to_string(err) + " \"" +       \
                               hipGetErrorString(err) + "\"");                 \
    }                                                                          \
  } while (0)

int main() {
  int *d_ptr{nullptr};
  HIP_CHECK(hipMalloc(&d_ptr, sizeof(int)));
  HIP_CHECK(hipMemset(d_ptr, 0, sizeof(int)));
  get_kernel()<<<1,1>>>(d_ptr);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
  int h_value{-2};
  HIP_CHECK(hipMemcpy(&h_value, d_ptr, sizeof(int), hipMemcpyKind::hipMemcpyDefault));
  printf("h_value = %d\n", h_value);
  assert(h_value == 2);
  HIP_CHECK(hipFree(d_ptr));
}
EOF

# "-fgpu-rdc", "-fno-gpu-rdc" or empty
RELOCATABLE_DEVICE_CODE_FLAG="-fgpu-rdc"
COMPILER="hipcc"

set -e
HIPCC_VERBOSE=1 ${COMPILER} -g ${RELOCATABLE_DEVICE_CODE_FLAG} -fPIC  -O0 -x hip -c library1.hip -o library1.o
HIPCC_VERBOSE=1 ${COMPILER} -g ${RELOCATABLE_DEVICE_CODE_FLAG} -fPIC  -O0 -x hip -c library2.hip -o library2.o
HIPCC_VERBOSE=1 ${COMPILER} ${RELOCATABLE_DEVICE_CODE_FLAG} --hip-link -shared -fPIC -o libdyn_lib.so library1.o library2.o

HIPCC_VERBOSE=1 ${COMPILER} -g ${RELOCATABLE_DEVICE_CODE_FLAG} -fPIC -O0 -x hip -c main.hip -o main.o
HIPCC_VERBOSE=1 ${COMPILER} -g ${RELOCATABLE_DEVICE_CODE_FLAG} --hip-link -L. -ldyn_lib -o main main.o
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${SCRIPT_DIR}/build  ./main
cd -