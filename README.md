# Maier-Saupe liquid crystal hydrodynamics

## Introduction
This repo comprises code written to simulate liquid crystal systems.
In particular, we use a field theory based on the Maier-Saupe free energy (which uses the mean-field approximation) to look at the dynamics of liquid crystals.
As opposed to the typical Landau-de Gennes free energy expression (essentially a Taylor-Series expansion of free energy), Maier-Saupe allows for liquid crystal systems which exhibit anisotropic elasticity.
Additionally, we hope to integrate hydrodynamics into our simulations.
We will use these simulations to investigate how anisotropy and hydrodynamics couple to affect various characteristics of these systems.

## Dependencies
Dependencies are as follows:

* deal.II
* Boost
* HighFive C++ Interface for HDF5
  - HDF5

The simulations that are run are, by and large, finite element simulations.
Because of this, we have based the project on the [deal.II finite element library](https://dealii.org/).
As such, this library is *absolutely necessary* to use any code in this repo.
The documentation for deal.II is extremely thorough, and they have many tutorials to help get started. 
As far as dependencies, deal.II has many optional dependencies which may or may not need to be installed depending on the scale and scope of the simulations.
For example, MPI is necessary if simulations are run on clusters. 

An external Boost library is also necessary for things like compiling the unit tests and reading in command-line arguments.
Deal.II is able to be built with an internal--somewhat slimmed down--version of Boost, but it is preferred here to have the full library.

Finally, HighFive allows an easy way to write data to HDF5 files.
It relies on the HDF5 library, so of course that is a dependency.

As far as installing these dependencies on your machine, [From the Ground Up](from-the-ground-up.md) provides notes on how a Ubuntu user installed these dependencies
and created a development environment for this software.
This should extend to most linux systems.

## Structure
Roughly following the Pitchfork Convention for code and just hap-hazard for other stuff.
See README's in each of these folders for further details:

- src

  *Directory defining libraries shared between executables*

- app

  *Source for various endpoint executables*
  - analysis
  - simulations
  - examples
  - theory

- test

  *Executables to check behavior of library*

- talks

  *Presentations about work done with this code*

- theory

  *Notes about theoretical backing of this code*