FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04

WORKDIR /usr/src/app

RUN apt update
RUN apt install -y python3.7-dev python3-pip
RUN update-alternatives --install $(which python3) python3 $(which python3.7) 1
RUN ln -s $(which python3) /usr/bin/python && update-alternatives --install $(which python) python $(which python3) 1
RUN ln -s $(which pip3) /usr/bin/pip && update-alternatives --install $(which pip) pip $(which pip3) 1
RUN pip install --upgrade pip

RUN apt install -y \
    build-essential cmake pkg-config git unzip wget \
    gcc-7 g++-7 \
    libjpeg-dev libtiff5-dev libpng-dev \
    ffmpeg libavcodec-dev libavformat-dev libswscale-dev libxvidcore-dev libx264-dev libxine2-dev \
    libv4l-dev v4l-utils \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    libgtk-3-dev \
    mesa-utils libgl1-mesa-dri libgtkgl2.0-dev libgtkglext1-dev \
    libatlas-base-dev gfortran libeigen3-dev

RUN \
    mkdir opencv && \
    cd opencv && \
    pip install numpy==1.20.2 && \
    wget -O opencv.zip https://github.com/opencv/opencv/archive/4.5.1.zip && unzip opencv.zip && \
    wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.5.1.zip && unzip opencv_contrib.zip && \
    cd opencv-4.5.1 && \
    mkdir build && cd build && \
    cmake \
    -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D PYTHON_DEFAULT_EXECUTABLE=$(which python3) \
    -D PYTHON_EXECUTABLE=$(which python) \
    -D PYTHON3_EXECUTABLE=$(which python3) \
    -D BUILD_NEW_PYTHON_SUPPORT=ON \
    -D BUILD_opencv_python3=ON \
    -D BUILD_opencv_python2=OFF \
    -D BUILD_PACKAGE=ON \
    -D WITH_GTK=ON \
    -D WITH_OPENGL=ON \
    -D WITH_CUDA=ON \
    -D WITH_CUFFT=ON \
    -D WITH_CUBLAS=ON \
    -D WITH_TBB=ON \
    -D WITH_IPP=ON \
    -D OPENCV_DNN_CUDA=ON \
    -D INSTALL_PYTHON_EXAMPLES=ON \
    -D ENABLE_FAST_MATH=1 \
    -D CUDA_FAST_MATH=1 \
    -D CMAKE_C_COMPILER=/usr/bin/gcc-7 \
    -D CMAKE_CXX_COMPILER=/usr/bin/g++-7 \
    -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-11.1 \
    -D PYTHON3_INCLUDE_DIR=/usr/include/python3.7m \
    -D PYTHON3_PACKAGES_PATH=/usr/local/lib/python3.7/dist-packages \
    -D PYTHON3_NUMPY_INCLUDE_DIRS=/usr/local/lib/python3.7/dist-packages/numpy/core/include \
    -D PYTHON3_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.7m.so \
    -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.5.1/modules .. && \
    make -j$(nproc --all) && \
    make install && \
    ldconfig -v && \
    cd /usr/src/app && rm -rvf opencv

RUN pip install \
    torch==1.8.1+cu111 \
    torchvision==0.9.1+cu111 \
    torchaudio==0.8.1 \
    -f https://download.pytorch.org/whl/torch_stable.html

COPY ./requirement.txt ./requirement.txt
RUN pip install -r requirement.txt

# Build jupyter
COPY ./scripts/build_jupyter.sh ./build_jupyter.sh
RUN ./build_jupyter.sh 8888
