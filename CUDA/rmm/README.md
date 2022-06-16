## Install 

My env:
- Ubuntu 20.04LTS
- GCC 9.4
- CUDA 11.3
- cmake 3.23.1


```sh
git clone --recurse-submodules https://github.com/rapidsai/rmm.git
cd rmm
git checkout branch-0.19 
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX 
make -j
make install
```


## Usage

### Compile

cmake
```
find_package(rmm)
target_link_libraries(${target} PRIVATE/PUBLIC rmm::rmm)
```


