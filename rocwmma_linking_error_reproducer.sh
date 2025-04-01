#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

mkdir -p ${SCRIPT_DIR}/build
rm ${SCRIPT_DIR}/build/*
cd ${SCRIPT_DIR}/build

cat > library.hip << EOF
#include <rocwmma/rocwmma.hpp>
EOF

cat > main.hip << EOF
#include <rocwmma/rocwmma.hpp>

int main() {}
EOF

set -xe
hipcc -D__HIP_ROCclr__=1 -fgpu-rdc -O3 -x hip --offload-arch=gfx90a -c library.hip -o library.o
hipcc -D__HIP_ROCclr__=1 -fgpu-rdc -O3 -x hip --offload-arch=gfx90a -c main.hip    -o main.o
hipcc --hip-link --offload-arch=gfx90a -fgpu-rdc -o main main.o library.o
cd -
