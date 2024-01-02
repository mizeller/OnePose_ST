FROM nvidia/cudagl:11.3.1-devel-ubuntu20.04

## INIT SETUP
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
RUN apt-get update && apt-get install -y \
    imagemagick \
    tree \
    git \
    cmake \
    vim \
    bzip2 \
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
    ffmpeg \
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

## INSTALL PYTHON + PACKAGES (basics + python3/pip)
RUN apt-get update && apt-get install -y \
    python3-flake8 \
    python3-opencv \
    python3-pip \
    python3-pytest-cov \
    python3-setuptools \
    && rm -rf /var/lib/apt/lists/*

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


## INSTALL ONEPOSE++ + DEPENDENCIES (python packages/submodules)
WORKDIR /OnePose_ST
COPY requirements.txt /OnePose_ST

RUN python3 -m pip install -r /OnePose_ST/requirements.txt
RUN python3 -m pip install imageio[ffmpeg]

COPY co-tracker/setup.py /OnePose_ST
RUN python3 -m pip install -e .

RUN ln -s /usr/bin/gcc-9 /usr/local/cuda-11.3/bin/gcc && ln -s /usr/bin/g++-9 /usr/local/cuda-11.3/bin/g++

# ------------------------------------ Start: DeepLM Stuff ---------------------------------- #
# WORKDIR /OnePose_ST/submodules/DeepLM

# # install DeepLM dependencies and test them
# RUN mkdir build && \
#     cd build && \
#     CUDACXX=/usr/local/cuda-11.3/bin/nvcc && \
#     # cmake .. -DCMAKE_BUILD_TYPE=Release -DWITH_CUDA=ON -DCMAKE_CXX_STANDARD=17 -DCMAKE_STANDARD_REQUIRED=ON && \
#     cmake .. -DCMAKE_BUILD_TYPE=Release -DWITH_CUDA=ON && \
#     make -j8 && \
#     export PYTHONPATH=$PYTHONPATH:$(pwd) && \
#     cd ..

# RUN mkdir -p data && \ 
#     cd data && \
#     wget https://grail.cs.washington.edu/projects/bal/data/ladybug/problem-49-7776-pre.txt.bz2 --no-check-certificate && \
#     bzip2 -d problem-49-7776-pre.txt.bz2 && \ 
#     cd ..

# #global lm solver
# RUN TORCH_USE_RTLD_GLOBAL=YES && \ 
#     python3 examples/BundleAdjuster/bundle_adjuster.py \
#     --balFile ./data/problem-49-7776-pre.txt \
#     --device cuda
#     RUN cp /OnePose_ST/backup/deeplm_init_backup.py /OnePose_ST/submodules/DeepLM/__init__.py
# ------------------------------------ End: DeepLM Stuff ------------------------------------ #

# TODO: somehow these wget calls always crash w/ this error: ERROR 429: TOO MANY REQUESTS.
# not sure how to resolve them...will just download the weights manually in the repo...

# RUN mkdir /OnePose_ST/weight && cd /OnePose_ST/weight 
# RUN wget --wait 10 --random-wait --continue https://zenodo.org/record/8086894/files/LoFTR_wsize9.ckpt?download=1 -O LoFTR_wsize9.ckpt
# RUN wget --wait 10 --random-wait --continue https://zenodo.org/record/8086894/files/OnePosePlus_model.ckpt?download=1 -O OnePosePlus_model.ckpt

# TODO: this downloads a pre-trained model - not of interest for now... 
# RUN mkdir -p /OnePose_ST/data/demo/sfm_model/outputs_softmax_loftr_loftr/
# WORKDIR /OnePose_ST/data/demo/sfm_model/outputs_softmax_loftr_loftr/
# RUN wget https://zenodo.org/record/8086894/files/SpotRobot_sfm_model.tar?download=1 -O SpotRobot_sfm_model.tar
# RUN tar -xvf SpotRobot_sfm_model.tar
# COPY ./sfm_model /OnePose_ST/data
# COPY ./data/SpotRobot /data/SpotRobot

# ------------------------------------ Start: ROS2 Stuff ------------------------------------ #
## SETUP ROS2 Foxy 
# TODO: check paths & make sure these files exist check out the spot_pose_estimation repo for reference

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

# RUN apt-get update \
#     && apt install ros-foxy-rmw-cyclonedds-cpp -y \
#     && rm -rf /var/lib/apt/lists/*

# RUN rosdep init && rosdep update

# RUN mkdir -p /ros2_ws/src

# WORKDIR /ros2_ws
# COPY ./ros2/spot_pose_estimation /ros2_ws/src/spot_pose_estimation
# RUN . /opt/ros/foxy/setup.sh \
#     && python3 -m pip install pytransform3d \
#     && colcon build --packages-select spot_pose_estimation
# ------------------------------------ End: ROS2 Stuff -------------------------------------- #