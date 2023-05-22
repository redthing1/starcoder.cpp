set -e
mkdir build && cd build
cmake ..
make -j$(nproc)