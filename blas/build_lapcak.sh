#!/bin/bash

set -e

INSTALL_DIR=./local

if [ ! -d ./lapack ]; then
    git clone https://github.com/Reference-LAPACK/lapack
fi

mkdir -p lapack-build

cd lapack-build
cmake -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR ../lapack && make
