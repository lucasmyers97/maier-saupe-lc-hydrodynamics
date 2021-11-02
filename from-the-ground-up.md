# From the Ground Up

Tom Eichlersmith
Notes on building this repository on a new computer.

# Operating System
I am running Ubuntu 20.04
```
$ cat /etc/os-release
NAME="Ubuntu"
VERSION="20.04.3 LTS (Focal Fossa)"
ID=ubuntu
ID_LIKE=debian
PRETTY_NAME="Ubuntu 20.04.3 LTS"
VERSION_ID="20.04"
```
**Note**: I have been using this laptop for development of other code,
so I may already have some of these dependencies available by accident.
For example, I already have `git` installed.

# Dependencies
Poked around the various executables and found the following dependencies.

- HDF5 (behind HighFive)
- Boost (for HighFive)
- HighFive
- Eigen
- Deal II
- CUDA

I am putting all the dependencies and their installations in a specific directory
to make later cleanup easier. Let's just call that directory `<deps>`
(for me, `<deps> == ~/lucas/`).

## HDF5
- Download source code from [website](https://www.hdfgroup.org/downloads/hdf5/source-code/)
  - I chose to download the tarball gzipped version
- Unpack the tarball
```
cd <deps>
tar xzvf hdf5-1.12.0.tar.gz
```
- Configure the build
```
cd hdf5-1.12.0
./configure --prefix=$PWD/install
```
- Build and Install (about an hour on a single core)
```
make install
```

## Boost
HighFive (and us) can use the version of Boost available in Ubuntu packages.
```
sudo apt install libboost-all-dev
```

## HighFive
- Download the `git` repository and checkout the latest release.
```
cd <deps>
git clone https://github.com/BlueBrain/HighFive
git checkout -b v2.3.1
```
- Follow instructions in README to build and install.
  **Note**: Since my HDF5 installation is not in a system path, I need to provide it to cmake.
```
cd HighFive
cmake -B build -S . \
  -DCMAKE_INSTALL_PREFIX=install \
  -DHIGHFIVE_EXAMPLES=OFF \
  -DHDF5_ROOT=<deps>/hdf5-1.12.0/install
cd build
make install
```

## Eigen
- Download source code from [website](https://eigen.tuxfamily.org/index.php?title=Main_Page)
```
cd <deps>
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
```
- Unpack it
```
tar xzvf eigen-3.4.0
```
- Configure the build
```
cd eigen-3.4.0
cmake -B build -S . -DCMAKE_INSTALL_PREFIX=install
```
- Build and Install
```
cd build
make install
```

## Deal II
The [Deal II Website](https://www.dealii.org/download.html) claims to have a working package in the Ubuntu repos,
but I wasn't able to get that to work, so I've put it in the `<deps>` direcotry with the other special dependencies.
This is annoying because Deal II is pretty large and takes many hours to compile on one core, but I will have to make do.
- Download source code
```
cd <deps>
wget https://github.com/dealii/dealii/releases/download/v9.3.1/dealii-9.3.1.tar.gz
```
- Unpack it
```
tar xzvf dealii-9.3.1.tar.gz
```
- Configure the build
```
cd dealii-9.3.1
cmake -B build -S . -DCMAKE_INSTALL_PREFIX=install
```
- Build and Install (takes many hours)
```
make install
```

**Note**: Deal II has a pretty good [page about using CMake](https://dealii.org/developer/users/cmake_user.html).

## CUDA
I don't have a CUDA-compatible GPU.
It looks like this [NVidia Blog Post](https://developer.nvidia.com/blog/building-cuda-applications-cmake/) may be helpful.

# Environment
Since I wanted to avoid installing these dependencies to a more permanent location,
I worte a simple environment script in the base directory `env.sh` which adds these installations to the necessary
environment variables when you provide the directory they all are in.

For example, I have all of these directories in the directory `~/lucas`, so I would
```
source env.sh ~/lucas
```
To setup the environment and then building this repository becomes slightly easier.
```
cmake -B build -S . -DCMAKE_INSTALL_PREFIX=install
cd build
make install
```

