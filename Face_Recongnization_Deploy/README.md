
# tensor Rt Dynamic shape , onnx model convert and infer demo

## Install Env：
https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-tar

```
opencv build Flag : -D WITH_CUDNN=OFF -D OPENCV_DNN_CUDA=ON
```


## Doc 
+ developer Guide: https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html
+ python API : https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/index.html
+ C/C++  API : https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/  
+ sample Doc : https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#python_samples_section
+ Jetson Infer Doc : https://github.com/dusty-nv/jetson-inference
> ref :  https://github.com/Star-Clouds/CenterFace

JetPack Download Addr: https://developer.nvidia.com/zh-cn/embedded/jetpack


### docker caffe
```
sudo apt-get update
sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
sudo apt-get update
sudo apt-get install docker-ce
sudo docker run hello-world   

sudo docker pull spellrun/caffe-cpu
sudo docker run  -it -v /home/:/mnt/  spellrun/caffe-cpu "/bin/bash"
sudo apt-get install libprotobuf-dev protobuf-compiler 

```
