FROM nvidia/cudagl:11.3.1-devel-ubuntu20.04

## INIT SETUP
# Prevent stop building ubuntu at time zone selection.  
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC

# Fundamentals
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    vim \
    build-essential \
    libmetis-dev \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libboost-test-dev \
    libeigen3-dev \
    libatlas-base-dev \
    libsuitesparse-dev \
    libfreeimage-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libcgal-qt5-dev \
    # additional packages hermann
    bash-completion \
    clang-format \
    curl \
    gnupg2 \
    locales \
    lsb-release \
    rsync \
    software-properties-common \
    wget \
    unzip \
    mlocate \
    && rm -rf /var/lib/apt/lists/*


## BUILD AND INSTALL CERES SOLVER
RUN git clone https://github.com/ceres-solver/ceres-solver.git --branch 1.14.0
RUN cd ceres-solver && \
    mkdir build && \
    cd build && \
    cmake .. -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF && \
    make -j4 && \
    make install

## BUILD AND INSTALL COLMAP
RUN git clone https://github.com/colmap/colmap.git --branch 3.7
RUN cd colmap && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make -j4 && \
    make install

## INSTALL PYTHON + PACKAGES (TODO)

# basics
RUN apt-get update && apt-get install -y \
    python3-flake8 \
    python3-opencv \
    python3-pip \
    python3-pytest-cov \
    python3-setuptools \
    && rm -rf /var/lib/apt/lists/*

# python3 (pip)
RUN alias pip="python3 -m pip"
RUN python3 -m pip install --upgrade pip && python3 -m pip install -U \
    argcomplete \
    autopep8 \
    flake8 \
    flake8-blind-except \
    flake8-builtins \
    flake8-class-newline \
    flake8-comprehensions \
    flake8-deprecated \
    flake8-docstrings \
    flake8-import-order \
    flake8-quotes \
    pytest-repeat \
    pytest-rerunfailures \
    pytest \
    pydocstyle \
    scikit-learn \
    loguru \
    h5py

## INSTALL ONEPOSE++ + DEPENDENCIES (TODO)

## INSTALL ROS2 + DEPENDENCIES (TODO)


# # Install cudnn8 and move necessary header files to cuda include directory
# RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt install libcudnn8-dev -y && \
#         cp /usr/include/cudnn_version.h /usr/local/cuda/include && \
#         cp /usr/include/cudnn.h /usr/local/cuda/include/ && \
#         rm -rf /var/lib/apt/lists/*




# # Setup ROS2 Foxy
# RUN locale-gen en_US en_US.UTF-8
# RUN update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
# ENV LANG=en_US.UTF-8

# RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
# RUN sh -c 'echo "deb [arch=$(dpkg --print-architecture)] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2-latest.list'

# RUN apt-get update && apt-get install -y \
#     python3-colcon-common-extensions \
#     python3-rosdep \
#     python3-vcstool \
#     ros-foxy-camera-calibration-parsers \
#     ros-foxy-camera-info-manager \
#     ros-foxy-ros-base \
#     ros-foxy-launch-testing-ament-cmake \
#     ros-foxy-v4l2-camera \
#     ros-foxy-vision-msgs \
#     ros-foxy-sensor-msgs-py \
#     ros-foxy-stereo-image-proc \
#     ros-foxy-pcl-ros \
#     ros-foxy-usb-cam \
#     && rm -rf /var/lib/apt/lists/*

# RUN apt-get update && apt-get install -y \
#     && apt install ros-foxy-rmw-cyclonedds-cpp -y \
#     && rm -rf /var/lib/apt/lists/*

# RUN rosdep init && rosdep update

# RUN mkdir -p /ros2_ws/src

# Install ROS2
# ENV LANG=en_US.UTF-8
# RUN apt update && apt install -y software-properties-common curl && add-apt-repository universe
# RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
# RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null
# RUN apt update && apt install -y ros-foxy-ros-base

# COPY ./OnePose_Plus_Plus_Spot /OnePose_Plus_Plus_Spot
# WORKDIR /OnePose_Plus_Plus_Spot
# RUN python3 -m pip install --ignore-installed -r /OnePose_Plus_Plus_Spot/requirements.txt \
#     && python3 -m pip uninstall -y torch torchvision numpy \
#     && pip freeze | grep cu12 | xargs pip uninstall -y \
#     && python3 -m pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 numpy==1.20.3 --extra-index-url https://download.pytorch.org/whl/cu113
# # RUN python3 -m pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
# # RUN python3 -c "import torch; print(torch.__version__); assert False"
# RUN ln -s /usr/bin/gcc-9 /usr/local/cuda-11.3/bin/gcc && ln -s /usr/bin/g++-9 /usr/local/cuda-11.3/bin/g++
# # ENV TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0 7.5 8.0 8.6 9.0+PTX"
# WORKDIR /OnePose_Plus_Plus_Spot/submodules/DeepLM
# # RUN mkdir build \
# #     && cd build \
# #     && CUDACXX=/usr/local/cuda-11.3/bin/nvcc cmake .. -DCMAKE_CXX_STANDARD=17 -DCMAKE_STANDARD_REQUIRED=ON  -DCMAKE_BUILD_TYPE=Release -DWITH_CUDA=ON \
# #     && make -j8 \
# #     && cd .. \
# #     && CUDACXX=/usr/local/cuda-11.3/bin/nvcc sh example.sh \
# #     && cp /OnePose_Plus_Plus_Spot/backup/deeplm_init_backup.py /OnePose_Plus_Plus_Spot/submodules/DeepLM/__init__.py
# RUN mkdir /OnePose_Plus_Plus_Spot/weight \
#     && cd /OnePose_Plus_Plus_Spot/weight \
#     && wget https://zenodo.org/record/8086894/files/LoFTR_wsize9.ckpt?download=1 -O LoFTR_wsize9.ckpt \
#     && wget https://zenodo.org/record/8086894/files/OnePosePlus_model.ckpt?download=1 -O OnePosePlus_model.ckpt
# RUN mkdir -p /OnePose_Plus_Plus_Spot/data/demo/sfm_model/outputs_softmax_loftr_loftr/
# WORKDIR /OnePose_Plus_Plus_Spot/data/demo/sfm_model/outputs_softmax_loftr_loftr/
# RUN wget https://zenodo.org/record/8086894/files/SpotRobot_sfm_model.tar?download=1 -O SpotRobot_sfm_model.tar
# RUN tar -xvf SpotRobot_sfm_model.tar
# COPY ./sfm_model /OnePose_Plus_Plus_Spot/data
# COPY ./data/SpotRobot /data/SpotRobot

# WORKDIR /ros2_ws
# COPY ./ros2/spot_pose_estimation /ros2_ws/src/spot_pose_estimation
# RUN . /opt/ros/foxy/setup.sh \
#         && python3 -m pip install pytransform3d \
#         && colcon build --packages-select spot_pose_estimation

# WORKDIR /OnePose_Plus_Plus_Spot
# COPY ./ros_entrypoint.sh /OnePose_Plus_Plus_Spot/ros_entrypoint.sh
# ENTRYPOINT ["/OnePose_Plus_Plus_Spot/ros_entrypoint.sh"]