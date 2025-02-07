#!/bin/bash
# This bash script is intended to be run when editing C++ code in the src directory.

if [ ! -d "build" ]; then
    mkdir build
fi

cd build
emcmake cmake ..
make
cd ..