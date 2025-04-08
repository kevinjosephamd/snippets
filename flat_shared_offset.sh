#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

mkdir -p ${SCRIPT_DIR}/build
rm ${SCRIPT_DIR}/build/*
cd ${SCRIPT_DIR}/build

cat > library_interface.h << EOF
#pragma once
#include<hip/hip_runtime.h>
#include <functional>
#include <iostream>
#include <memory>

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

void entrypoint();
EOF

cat > library1.hip << EOF
#include "library_interface.h"

__device__ void scrible_to_shared_memory(uint32_t start_offset);
__device__ uint32_t flat_to_shared_offset(int*  ptr);

__global__ void kernel(int* out) {
    extern __shared__ int shared[];
    auto offset = flat_to_shared_offset(shared);
    printf("kernel %p\n", shared);
    scrible_to_shared_memory(offset);
    __syncthreads();
    out[threadIdx.x] = shared[threadIdx.x];
    __syncthreads();
}


void entrypoint() {
    int *d_ptr{nullptr};
    int block_size = 128;
    HIP_CHECK(hipMalloc(&d_ptr, sizeof(int) * block_size));
    kernel<<<1,block_size, block_size * sizeof(int)>>>(d_ptr);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<int> h_data;
    h_data.resize(block_size);
    HIP_CHECK(hipMemcpy(h_data.data(), d_ptr, sizeof(int) * block_size, hipMemcpyKind::hipMemcpyDefault));
    for(auto const& entry : h_data) {
        std::cout << entry << ' ';
    }
    std::cout << '\n';
}

EOF

cat > library2.hip << EOF
#include "library_interface.h"

__device__ int* shared_offset_to_flat(uint32_t offset);

__device__ void scrible_to_shared_memory(uint32_t start_offset) {
    int* ptr = shared_offset_to_flat(start_offset);
    ptr[threadIdx.x] = threadIdx.x;
    printf("scrible_to_shared_memory %p %u %d \n", ptr, start_offset, ptr[threadIdx.x]);
}

EOF

cat > library3.hip << EOF
#include "library_interface.h"

__device__ uint64_t shared_base_offset() {
  uint32_t lo, hi;
  asm volatile(
      // Move the symbol address into s[6:7], then copy those to lo/hi
      "s_mov_b64 s[6:7], src_shared_base \n" // https://github.com/ROCm/llvm-project/blob/1c0f91c6fd453395b24f2ecefd1f843a99f1da9f/llvm/docs/AMDGPUOperandSyntax.rst#ival
      "s_mov_b32 %0, s6 \n"
      "s_mov_b32 %1, s7 \n"
      : "=s"(lo), "=s"(hi)     // outputs (two SGPRs)
      :                        // no inputs
      : "s6", "s7"             // clobbers
  );
  return (static_cast<uint64_t>(hi) << 32) | lo;
}

__device__ uint32_t flat_to_shared_offset(int* ptr)
{
  union { int* p; uint64_t i; } uptr;
  uptr.p = ptr;
  return static_cast<uint32_t>(uptr.i - shared_base_offset());
}

__device__ int* shared_offset_to_flat(uint32_t offset)
{
  union { int* p; uint64_t i; } uptr;
  uptr.i = shared_base_offset() + offset;
  return uptr.p;
}
EOF

cat > main.hip << EOF
#include "library_interface.h"

int main() {
  entrypoint();  
}
EOF

# "-fgpu-rdc", "-fno-gpu-rdc" or empty
RELOCATABLE_DEVICE_CODE_FLAG="-fgpu-rdc"
COMPILER="hipcc"

set -e
HIPCC_VERBOSE=1 ${COMPILER} -g ${RELOCATABLE_DEVICE_CODE_FLAG} -fPIC  -O0 -x hip -c library1.hip -o library1.o
HIPCC_VERBOSE=1 ${COMPILER} -g ${RELOCATABLE_DEVICE_CODE_FLAG} -fPIC  -O0 -x hip -c library2.hip -o library2.o
HIPCC_VERBOSE=1 ${COMPILER} -g ${RELOCATABLE_DEVICE_CODE_FLAG} -fPIC  -O0 -x hip -c library3.hip -o library3.o
HIPCC_VERBOSE=1 ${COMPILER} ${RELOCATABLE_DEVICE_CODE_FLAG} --hip-link -shared -fPIC -o libdyn_lib.so library1.o library2.o  library3.o

HIPCC_VERBOSE=1 ${COMPILER} -g ${RELOCATABLE_DEVICE_CODE_FLAG} -fPIC -O0 -x hip -c main.hip -o main.o
HIPCC_VERBOSE=1 ${COMPILER} -g ${RELOCATABLE_DEVICE_CODE_FLAG} --hip-link -L. -ldyn_lib -o main main.o
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${SCRIPT_DIR}/build  ./main
cd -
