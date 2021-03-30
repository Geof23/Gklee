FROM ubuntu:18.04

MAINTAINER Tanmay

# Setting Environment variables.
ENV KLEE_HOME_DIR=/home/Gklee
ENV PATH=$KLEE_HOME_DIR/bin:$PATH

# Updating apt-get distro
RUN apt-get update

# Installing all dependencies using apt-get
RUN apt-get install -y gcc-4.8 g++-4.8 cmake git wget python vim doxygen zlib1g zlib1g-dev bison flex build-essential

# Switching alternatives to gcc and g++ to 4.8
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.8 10
RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.8 10

# Cloning the repository
RUN cd /home && git clone https://github.com/tanmaytirpankar/Gklee.git

# Fixing gklee-nvcc to use the correct gcc toolchain
RUN cd /home/Gklee && sed -i "s|-Xclang -fgpu-device|-Xclang -fgpu-device -gcc-toolchain /usr/include/c++/4.8.5|g" bin/gklee-nvcc

# Changing build mode to Release
RUN cd /home/Gklee && sed -i "s|CMAKE_BUILD_TYPE RelWithDebInfo|CMAKE_BUILD_TYPE Release|g" CMakeLists.txt

# Building
RUN cd /home/Gklee && mkdir build && cd build && cmake .. && make
