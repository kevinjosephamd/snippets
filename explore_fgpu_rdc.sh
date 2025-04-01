#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

mkdir -p ${SCRIPT_DIR}/build
rm ${SCRIPT_DIR}/build/*
cd ${SCRIPT_DIR}/build

cat > library_interface.h << EOF
#pragma once

template<typename T>
__global__ void my_kernel(T* inout);

__device__ void no_op();

template<typename T>
using kernel_t = decltype(&my_kernel<T>);

template<typename T>
auto get_kernel() -> kernel_t<T>;

extern template kernel_t<int> get_kernel<int>();
EOF

cat > library1.hip << EOF
#include<hip/hip_runtime.h>
#include "library_interface.h"
template<typename T>
auto get_kernel() -> kernel_t<T> {
  return my_kernel<T>;
}

__device__ void no_op() {}

template kernel_t<int> get_kernel<int>();
EOF

cat > library2.hip << EOF
#include<hip/hip_runtime.h>
#include "library_interface.h"

template<typename T>
__global__ void my_kernel(T* inout) {
  no_op();
  printf("Hello from my_kernel\n");
  *inout = 1;
}

template __global__ void my_kernel<int>(int* inout);
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
  auto kernel = get_kernel<int>();
  int *d_ptr{nullptr};
  HIP_CHECK(hipMalloc(&d_ptr, sizeof(int)));
  HIP_CHECK(hipMemset(d_ptr, 0, sizeof(int)));
  kernel<<<1,1>>>(d_ptr);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
  int h_value{-2};
  HIP_CHECK(hipMemcpy(&h_value, d_ptr, sizeof(int), hipMemcpyKind::hipMemcpyDefault));
  printf("h_value = %d\n", h_value);
  assert(h_value == 1);
  HIP_CHECK(hipFree(d_ptr));
}
EOF

# "-fgpu-rdc" or empty
RELOCATABLE_DEVICE_CODE_FLAG="-fgpu-rdc"
# "/opt/rocm/llvm/bin/clang++" OR "hipcc"
COMPILER="/opt/rocm/llvm/bin/clang++"

set -xe
HIPCC_VERBOSE=1 ${COMPILER} -g ${RELOCATABLE_DEVICE_CODE_FLAG} -fPIC  -D__HIP_ROCclr__=1 --offload-arch=gfx90a -O0 -x hip -c library1.hip -o library1.o
HIPCC_VERBOSE=1 ${COMPILER} -g ${RELOCATABLE_DEVICE_CODE_FLAG} -fPIC  -D__HIP_ROCclr__=1 --offload-arch=gfx90a -O0 -x hip -c library2.hip -o library2.o
ar rcs libstat_lib.a library1.o library2.o
HIPCC_VERBOSE=1 ${COMPILER} -g ${RELOCATABLE_DEVICE_CODE_FLAG} -fPIC -D__HIP_ROCclr__=1 --offload-arch=gfx90a -O0 -x hip -c main.hip -o main.o
HIPCC_VERBOSE=1 ${COMPILER} -g ${RELOCATABLE_DEVICE_CODE_FLAG} --offload-arch=gfx90a --hip-link -L. -lstat_lib -o main main.o
./main
cd -