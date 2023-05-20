#!/usr/bin/env bash
mkdir temp
cd temp
cmake .. -DCMAKE_INSTALL_PREFIX=./../
make
make install
cd ..
rm -rf temp