#!/bin/bash
set -e
cd "$(dirname "$0")"

echo "=== 编译 jacobi ==="
g++ -std=c++17 -O2 jacobi.cpp -o jacobi

echo "=== 运行 jacobi N=64 ==="
./jacobi 64

echo ""
echo "=== 运行 jacobi N=128 (观察迭代次数增长) ==="
./jacobi 128
