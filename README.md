﻿# Maier-Saupe liquid crystal hydrodynamics [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14872508.svg)](https://doi.org/10.5281/zenodo.14872508)

## Introduction

This code is a massively parallel finite element simulation which models nematic liquid crystal dynamics based on the Ball-Majumdar Singular Potential method.
This models non-equilibrium nematic liquid crystal fields with a $3\times 3$ $Q$ tensor, so that topological defects may be simulated without additional consideration.
Because the singular potential method is used, anisotropy in the bend and splay elastic modes of the nematic may be simulated without further consideration (as opposed to the Lanuda-de Gennes $Q$ tensor model).
The most current executable is `NematicSystemMPISim`, and it is run by passing in a parameter file, which follows the [.toml](https://toml.io/en/) format.
Examples which have been used to run simulations for the "Numerical study of chiral ground states in a nematic liquid crystal confined to a cylinder with homeotropic anchoring" paper are available in the [chiral-nematics-supercomputer](parameter-files/chiral-nematics-supercomputer) folder.
Older parameter files may cause errors. 
The repo also contains theoretical notes on liquid crystal theory, as well as post-processing scripts and automatic code generation scripts. 
We give an overview of the structure below.
See [installation.md](installation.md) for details on how the repository may be installed on a Linux system.

## Structure
Roughly following the Pitchfork Convention for code and just hap-hazard for other stuff.
See README's in each of these folders for further details:

- app

  *Source for various endpoint executables*
  - analysis
    - Scripts for automatic code generation
    - Scripts for generating paper figures
    - Programmable filters for data analysis in Paraview
    - Source files for inverting Singular Potential outside of main simulation
    - General utilities for tensor calculus and nematic calculations
  - examples
    - Executables which demonstrate isolated functionality of various code pieces
  - simulations
    - All nematic simulations
    - Most recent is NematicSystemMPISim.cpp
  - theory

- parameter-files

    *Collection of parameter files used to run nematic simulations*
    - All `.prm` parameter files are outdated.
    - Most updated parameter files are in `chiral-nematics-supercomputer`

- src

  *Directory defining libraries shared between executables*


- test

  *Executables to check behavior of library*

- talks

  *Presentations about work done with this code*

- theory

  *Notes about theoretical backing of this code*
