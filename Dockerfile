# FROM geki/colmap
# FROM colmap/colmap:3.6-dev.3
# FROM colmap/colmap:latest

# trying to compile new Dockerimage from scratch
# basically combining the Dockerfile from the current COLMAP master and
# the one from spot_pose_estimation

######################################## COLMAP Dockerfile ########################################
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

ARG COLMAP_GIT_COMMIT=main
# NOTE: modified this from native to 86
ARG CUDA_ARCHITECTURES=86
ENV QT_XCB_GL_INTEGRATION=xcb_egl

# Prevent stop building ubuntu at time zone selection.  
ENV DEBIAN_FRONTEND=noninteractive

# Prepare and empty machine for building.
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    ninja-build \
    build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libsqlite3-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libceres-dev

# Build and install COLMAP.
RUN git clone https://github.com/colmap/colmap.git
RUN cd colmap && \
    git fetch https://github.com/colmap/colmap.git ${COLMAP_GIT_COMMIT} && \
    git checkout FETCH_HEAD && \
    mkdir build && \
    cd build && \
    cmake .. -GNinja -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} && \
    ninja && \
    ninja install && \
    cd .. && rm -rf colmap

######################################## COLMAP Dockerfile ########################################