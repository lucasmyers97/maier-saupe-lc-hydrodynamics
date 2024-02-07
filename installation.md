# Installation

Notes on how to get this repository installed.
*Note*: I have already been using this machine to actively develop this repo, so I may have missed some dependency that I already had installed.

# Operating System

I am running Ubuntu 22.04
``` bash
$ cat /etc/os-release
PRETTY_NAME="Ubuntu 22.04.3 LTS"
NAME="Ubuntu"
VERSION_ID="22.04"
VERSION="22.04.3 LTS (Jammy Jellyfish)"
VERSION_CODENAME=jammy
```

# Dependencies

The repo depends on the following libraries:

- MPI
- HDF5 (with MPI)
- P4EST
- Boost
- Trilinos
- Deal.II

## MPI

This is available via `apt`:
``` bash
sudo apt install openmpi-bin openmpi-common
```

## HDF5

This is also available via apt:
``` bash
sudo apt install hdf5-helpers hdf5-tools
```

## P4EST

Go to their website [here](https://p4est.org/) and get the latest release tarball (there should be a link on their page).
Otherwise, you can find all releases [here](https://github.com/p4est/p4est.github.io/tree/master/release/).
``` bash
curl https://p4est.github.io/release/p4est-tarball-name.tar.gz -o p4est-tarball-name.tar.gz
```
`deali.II` has helpful advice on installing this, found [here](https://dealii.org/current/external-libs/p4est.html).
In short, download the `deal.II` script found [here](https://dealii.org/current/external-libs/p4est-setup.sh) and execute:
``` bash
curl https://dealii.org/current/external-libs/p4est-setup.sh -o p4est-setup.sh
```
``` bash
chmod u+x p4est-setup.sh
./p4est-setup.sh p4est-x-y-z.tar.gz /path/to/installation
```

## Boost

Available via apt:
``` bash
sudo apt install libboost-dev
```

## Trilinos

Deal.II has some explanation for what's required [here](https://dealii.org/current/external-libs/trilinos.html)
To get the source, clone the git repository:
``` bash
git clone https://github.com/trilinos/Trilinos.git
```
``` bash
git checkout trilinos-release-13-4-branch
```
To build and install, the following commands work:
``` bash
cd trilinos-x.y.z
mkdir build
cd build

cmake                                                     \
-DTrilinos_ENABLE_Amesos=ON                               \
-DTrilinos_ENABLE_Epetra=ON                               \
-DTrilinos_ENABLE_EpetraExt=ON                            \
-DTrilinos_ENABLE_Ifpack=ON                               \
-DTrilinos_ENABLE_AztecOO=ON                              \
-DTrilinos_ENABLE_Sacado=ON                               \
-DTrilinos_ENABLE_SEACAS=OFF                              \
-DTrilinos_ENABLE_Teuchos=ON                              \
-DTrilinos_ENABLE_MueLu=ON                                \
-DTrilinos_ENABLE_ML=ON                                   \
-DTrilinos_ENABLE_ROL=ON                                  \
-DTrilinos_ENABLE_Tpetra=ON                               \
-DTrilinos_ENABLE_COMPLEX_DOUBLE=ON                       \
-DTrilinos_ENABLE_COMPLEX_FLOAT=ON                        \
-DTrilinos_ENABLE_Zoltan=ON                               \
-DTrilinos_VERBOSE_CONFIGURE=OFF                          \
-DTPL_ENABLE_MPI=ON                                       \
-DBUILD_SHARED_LIBS=ON                                    \
-DCMAKE_VERBOSE_MAKEFILE=OFF                              \
-DCMAKE_BUILD_TYPE=RELEASE                                \
-DCMAKE_INSTALL_PREFIX:PATH=path/to/trilinos/installation \
../

make install -jN
```
where `N` is the number of processors you wish to devote to compilation.
This takes a while.
I turned off the Seacas library because it was giving me dependency errors, and because Deal.II does not need it (except for grid reading, which we do not do).

## Deal.II

Clone the GitHub repo and `cd` to that directory:
``` bash
git clone https://github.com/dealii/dealii.git
cd dealii
```
Check which branches are available:
``` bash
git branch -a
```
The output should be something like:
``` bash
* master
  remotes/origin/HEAD -> origin/master
  remotes/origin/dealii-8.0
  remotes/origin/dealii-8.1
  remotes/origin/dealii-8.2
  remotes/origin/dealii-8.3
  remotes/origin/dealii-8.4
  remotes/origin/dealii-8.5
  remotes/origin/dealii-9.0
  remotes/origin/dealii-9.1
  remotes/origin/dealii-9.2
  remotes/origin/dealii-9.3
  remotes/origin/dealii-9.4
  remotes/origin/dealii-9.5
  remotes/origin/master
```
Here the most recent version is `dealii-9.5`.
Checkout the most recent version with:
``` bash
git checkout dealii-x.y
```
with `dealii-x.y` the most recent branch.
Make a build directory:
``` bash
mkdir -p build/dealii-x.y
cd build/dealii-x.y
```
The following command should work:
``` bash
cmake -DCMAKE_INSTALL_PREFIX=/path/to/install \
      -DDEAL_II_WITH_HDF5=ON \
      -DDEAL_II_WITH_KOKKOS=OFF \
      -DDEAL_II_WITH_MPI=ON \
      -DDEAL_II_WITH_P4EST=ON \
      -DDEAL_II_WITH_TRILINOS=ON \
      -DBOOST_DIR=/path/to/boost/install \
      -DTRILINOS_DIR=/path/to/trilinos/install \
      -DHDF5_DIR=/path/to/hdf5/install \
      -DP4EST_DIR=/path/to/p4est/install
      ../..
```
Finally, you can compile with:
``` bash
make -jN
```
where `N` is the number of processors you want to devote to compilation.

# Installing this library

First make a build directory:
``` bash
mkdir build
cd build
```
Then setup cmake:
``` bash
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/path/to/library/install \
      -Ddeal.II_DIR=/path/to/dealii \
      ../..
```
Finally, build and install:
``` bash
cd ..
cmake --build build --target install -- -jN
```
where `N` is the number of processors you want to devote to compilation.
