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


void library_function();

struct OuterContainer {
  struct Container {
    template<typename Callable>
    Container(Callable && callable) : m_callable(std::move(callable)) {}

    void invoke_callable() {
      m_callable();
    }

    std::function<int()> m_callable;
  };

  template<typename Callable>
  OuterContainer(Callable && callable) : inner_container(std::make_shared<Container>(std::move(callable))) {}

  void invoke_callable() {
    inner_container->invoke_callable();
  }

  std::shared_ptr<Container> inner_container;
};


EOF

cat > library1.hip << EOF
#include "library_interface.h"

extern OuterContainer g_container;

void library_function() {
  g_container.invoke_callable();
}

EOF

cat > library2.hip << EOF
#include "library_interface.h"

using kernel_type = __global__ void(*)();
auto get_kernel_ptr()-> kernel_type;

OuterContainer g_container([]()->int{
    get_kernel_ptr()<<<1,1>>>();
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    return 0;
});
EOF

cat > library3.hip << EOF
#include "library_interface.h"

__global__ void kernel() {
  printf("Hello from the GPU\n");
}

using kernel_type = __global__ void(*)();

kernel_type get_kernel_ptr() {
  return kernel;
}

EOF

cat > main.hip << EOF
#include "library_interface.h"

int main() {
  library_function();  
}
EOF

# "-fgpu-rdc", "-fno-gpu-rdc" or empty
RELOCATABLE_DEVICE_CODE_FLAG=""
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