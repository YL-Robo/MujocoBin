#!/bin/bash

if [ ! -d "orocos_kinematics_dynamics" ]; then
    git clone https://github.com/orocos/orocos_kinematics_dynamics.git
    cd orocos_kinematics_dynamics
    cd orocos_kdl
    mkdir build
    cd build
    cmake ..
    make -j$(nproc)
    sudo make install
    cd ../..
    cd python_orocos_kdl
    mkdir build
    cd build
    cmake ..
    make -j$(nproc)
    sudo make install
    cp PyKDL.so* ../../../.venv/lib/python3.10/site-packages/
fi

rm -rf orocos_kinematics_dynamics

if [ ! -d "kdl_parser" ]; then
    git clone git@github.com:jvytee/kdl_parser.git
    cd kdl_parser
    uv pip install .
fi

rm -rf kdl_parser