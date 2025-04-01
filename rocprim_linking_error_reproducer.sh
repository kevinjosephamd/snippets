#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

mkdir -p ${SCRIPT_DIR}/build
rm ${SCRIPT_DIR}/build/*
cd ${SCRIPT_DIR}/build

# Source for the first object file with a relocatable device function.
cat > library.hip << EOF
#include <rocprim/rocprim.hpp>

__device__ inline int inline_function(int) {
  return 0;
}

__device__ void ldg(float& x, const float* addr)
{
  x = rocprim::thread_load<rocprim::cache_load_modifier::load_ldg>(const_cast<float*>(addr));
}
EOF

# Source file that compiles to a second object file but references a device function in a different object file.
cat > main.hip << EOF
#include <hip/hip_runtime.h>
#include <rocprim/rocprim.hpp>

extern __device__ void ldg(float &x, const float *addr);

__device__ inline int inline_function(int) {
  return 0;
}

__global__ void kernel(float *ptr) {
  float x;
  ldg(x, const_cast<const float*>(ptr));
  x += rocprim::thread_load<rocprim::cache_load_modifier::load_ldg>(const_cast<float*>(ptr));
  *ptr = x;
}

int main() {
  float *d_ptr;
  hipMalloc(&d_ptr, sizeof(float));
  hipMemset(d_ptr, 0, sizeof(float));
  kernel<<<1,1>>>(d_ptr);
  hipDeviceSynchronize();
}
EOF

set -xe
hipcc -D__HIP_ROCclr__=1 -fgpu-rdc -O3 -x hip --offload-arch=gfx90a -c library.hip -o library.o
hipcc -D__HIP_ROCclr__=1 -fgpu-rdc -O3 -x hip --offload-arch=gfx90a -c main.hip    -o main.o
hipcc --hip-link --offload-arch=gfx90a -fgpu-rdc -o main main.o library.o
cd -
