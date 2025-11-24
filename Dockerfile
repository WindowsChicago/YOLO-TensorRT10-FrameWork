FROM nvcr.io/nvidia/tensorrt:24.08-py3

RUN apt-get -y update
RUN apt-get -y install build-essential cmake pkg-config unzip yasm git checkinstall \
    libjpeg-dev libpng-dev libtiff-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    libxvidcore-dev libx264-dev libmp3lame-dev libopus-dev \
    libmp3lame-dev libvorbis-dev \
    ffmpeg

RUN wget -O opencv.zip https://github.com/opencv/opencv/archive/refs/tags/4.10.0.zip && \
    wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/refs/tags/4.10.0.zip && \
    unzip opencv.zip && \
    unzip opencv_contrib.zip


RUN cd opencv-4.10.0 && mkdir -p build && cd build && \
cmake -DCMAKE_BUILD_TYPE=RELEASE \
-DCMAKE_INSTALL_PREFIX=/usr/local \
-DENABLE_FAST_MATH=1 \
-DCUDA_FAST_MATH=1 \
-DWITH_CUBLAS=1 \
-DWITH_CUDA=ON \
-DBUILD_opencv_cudacodec=OFF \
-DWITH_CUDNN=ON \
-DOPENCV_DNN_CUDA=ON \
-DCUDA_ARCH_BIN=7.5 \
-DWITH_V4L=ON \
-DWITH_QT=OFF \
-DWITH_GSTREAMER=ON \
-DOPENCV_GENERATE_PKGCONFIG=ON \
-DOPENCV_PC_FILE_NAME=opencv.pc \
-DOPENCV_ENABLE_NONFREE=ON \
-DOPENCV_EXTRA_MODULES_PATH=/workspace/opencv_contrib-4.10.0/modules \
-DINSTALL_PYTHON_EXAMPLES=OFF \
-DINSTALL_C_EXAMPLES=OFF \
-DBUILD_EXAMPLES=OFF .. && \
make -j4 && \
make install

WORKDIR /workspace/yolofast