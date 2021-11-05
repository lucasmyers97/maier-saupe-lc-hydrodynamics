#!/bin/bash

# attach necessary directories to various environment variables to help with development
__attach() {
  local __install_d="$(realpath $1)"

  if [[ ! -d ${__install_d} ]]; then
    return 1
  fi

  if [[ -z $PATH ]]; then
    export PATH=${__install_d}/bin
  else
    export PATH=${__install_d}/bin:${PATH}
  fi

  if [[ -z $LD_LIBRARY_PATH ]]; then
    export LD_LIBRARY_PATH=${__install_d}/lib
  else
    export LD_LIBRARY_PATH=${__install_d}/lib:${LD_LIBRARY_PATH}
  fi

  if [[ -z $CMAKE_PREFIX_PATH ]]; then
    export CMAKE_PREFIX_PATH=${__install_d}
  else
    export CMAKE_PREFIX_PATH=${__install_d}:${CMAKE_PREFIX_PATH}
  fi

  return 0
}

__main() {
  local __dep_home="$1"
  if [[ -z ${__dep_home} ]]; then
    echo "ERROR: env.sh requires the full path to the project directory."
    echo "       source env.sh /full/path/to/dir"
    return 1
  fi
  for dep in dealii-9.3.1 eigen-3.4.0 hdf5-1.12.0 HighFive maier-saupe-lc-hydrodynamics; do
    __attach ${__dep_home}/${dep}/install
  done
}

__main $@
