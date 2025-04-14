#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

mkdir -p ${SCRIPT_DIR}/build
rm ${SCRIPT_DIR}/build/*
cd ${SCRIPT_DIR}/build

cat > main.hip << EOF
#include<hip/hip_runtime.h>
#include<numeric>
#include<vector>
#include<iostream>

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

__global__ void kernel(int* d_array, int size) {
  int local = threadIdx.x >= size ? 0 : d_array[threadIdx.x];
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    local += __shfl_down_sync(__activemask(), local, offset);
  }
  if(threadIdx.x % warpSize == 0) {
    d_array[0] = local;
  }
}


int main() {
  constexpr int SIZE = 63;
  
  int *d_data{nullptr};
  HIP_CHECK(hipMalloc(&d_data, sizeof(int) * SIZE));
  
  std::vector<int> h_data(SIZE);
  std::iota(h_data.begin(), h_data.end(), 0);
  HIP_CHECK(hipMemcpy(d_data, h_data.data(), h_data.size() * sizeof(int), hipMemcpyKind::hipMemcpyHostToDevice));

  kernel<<<1, SIZE>>>(d_data, SIZE);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  int h_sum{-1};
  HIP_CHECK(hipMemcpy(&h_sum, d_data, sizeof(int), hipMemcpyKind::hipMemcpyDeviceToHost));

  std::cout << "Sum:" << h_sum << "\n";
  HIP_CHECK(hipFree(d_data));  
}
EOF

# "-fgpu-rdc", "-fno-gpu-rdc" or empty
RELOCATABLE_DEVICE_CODE_FLAG="-fgpu-rdc"
COMPILER="hipcc"

set -e
HIPCC_VERBOSE=1 ${COMPILER} -g ${RELOCATABLE_DEVICE_CODE_FLAG} -fPIC -O0 -DHIP_ENABLE_WARP_SYNC_BUILTINS -x hip -c main.hip -o main.o
HIPCC_VERBOSE=1 ${COMPILER} -g ${RELOCATABLE_DEVICE_CODE_FLAG} --hip-link -o main main.o
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${SCRIPT_DIR}/build  ./main
cd -