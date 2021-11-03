# maier-saupe-lc-hydrodynamics

## Dependencies
[From the Ground Up](from-the-ground-up.md) provides notes on how a Ubuntu user installed these dependencies
and created a development environment for this software.

- Boost
- HDF5
- HighFive C++ Interface for HDF5
- Deal II
- Eigen3
- CUDA?

## Structure
Roughly following the Pitchfork Convention for code and just hap-hazard for other stuff.

- src
  - Directory defining libraries shared between executables
- app
  - Various endpoint executables
- test
  - Executables to check behavior of library
- talks
  - Presentations about work done with this code
- theory
  - Notes about theoretical backing of this code
