# install

cd ~

git clone https://github.com/morgan322/ai-engine.git

git submodule init

git submodule update

# build

docker build -t ai-engine-image -f docker/Dockerfile .

docker run -itd --name=devel --privileged  --net=host --gpus=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video -v ~/ai-engine:/workspace/ ai-engine-image

mkdir build && cd build

cmake .. && make -j8
