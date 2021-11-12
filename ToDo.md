# To Do

## Physical Tests
-----------------
* Run for uniform configurations
  - run with different Kappa values
  - run with different starting configurations (isotropic, nematic, sometimes both)
* Run system with nematic director oscillating a small amount
* ~~Compute bulk free energy for nematic~~

## New Code
-----------
* Try Neumann boundary conditions with nematic
* Try LdG free energy with nematic
* Redefine degrees of freedom to matcch with Cody's
* Try with just eta, mu, nu

## Refactor Code
----------------
* ~~Make `read-defect-print-to-grid` take in directory for where to read/write data.~~
* Refactor `LagrangeMultiplier.cpp`
  - ~~Make it more efficient so that it only has to calculate exponentials once~~
  - ~~Document each method in the header file~~
  - Add unit-tests for each method
* Write scripts which call simulations and input parameters
* Test liquid crystal post-processor
* Write function which generates LC configurations
* Test function which generates LC configurations
* Figure out how to get `output_cody_data` to work at some point.

## Reorganize Code
------------------
* Write env.sh so that it can find packages so long as they are in *an* installation directory
* Rewrite CMakeLists.txt files so that we can find packages even if they were not installed with cmake
* Put classes which generate nematic configurations in their own files
* Put classes which output director field in their own files
* Figure out how to organize them -- maybe a big include file?
* Could make one class which is just a template (which has different instantiations based on type)
* Put boundary condition functions in separate file

## Documentation
----------------
* ~~Download Doxygen~~
* ~~Write Doxygen documentation for one source file~~
* ~~Write documentation for LagrangeMultiplier class~~
* ~~Link docs to a GitHub pages site~~
* Update all README's so that they give a good idea of what's going on
* For all analysis scripts, add a little blurb at the top which discusses what it does
* Include scripts and executables in Doxygen documentation (somehow)
* Modify from-the-ground-up.md to include dealii dependencies (also get rid of Eigen)
#### _Making Doxygen site useful_
* Figure out how to add front page, and what should go on front page.
* Get rid of "Files" on the sidebar
* Add examples with explanations to get people started
* Document simulations for people to look at

## Supercomputer
----------------
* Get logged on
* run simple c++ program
* figure out how to submit interactive job
* install or use dealii
* install or use cuda
* run simulation to see how long it takes

## GPU instantiation of LagrangeMultiplier
------------------------------------------
* ~~Test CUDA compilation with vector addition~~
* ~~Test `LU_Matrix` batched inversion in CUDA~~
* ~~Write kernel to generate Residual and Jacobian~~
* ~~Write program which iterates Newton's method to solve~~
* ~~Need to add #pragma unroll commands~~
* Invert shape functions
* Play with `__constant__` memory

## Old ToDo
-----------
* ~~Install boost~~
* ~~Figure out how to construct an empty vector~~
* ~~Write LagrangeMultiplier member variables as std::vectors of points~~
* ~~Make LagrangeMultiplier invert Q~~
* ~~Fix sign error in equation of motion~~
* ~~Write up isotropic problem~~
* ~~Take Gateaux derivative of equation of motion~~
* Implement Newton's method using dealii
  - ~~Make grid~~
  - ~~Distribute DOFS~~
  - ~~Write boundary-values function~~
  - ~~Write `setup_system` function~~
    - ~~Introduce `system_update`, `current_system`~~
  - ~~Populate matrix~~
    - ~~Return Lambda evaluated at quadrature points~~
    - ~~Return Jacobian evaluated at quadrature points, solve matrix equation with shape function rhs~~
  - ~~Populate rhs~~
  - ~~Remove hanging nodes, apply zero boundary condition to Newton Update~~
  - ~~Find matrix solver appropriate for the problem~~ UMFPACK Direct Solver
  - ~~Set boundary values for actual solution~~
  - ~~Compute the residual~~
  - Figure out how to set step size
  - ~~Ouput results (have this in the other file)~~
* Learn to use ParaView
  - ~~Display 2D liquid crystal configurations in ParaView~~
    - ~~Create function which returns Q-tensor for uniaxial configuration~~
    - ~~Project that function onto the finite element space~~
    - ~~Write DataPostprocessor function that gives the nematic configuration~~
    - ~~Write the nematic configuration to a .vtu file~~
    - ~~Open it in Paraview~~
    - ~~See if you can make the nematic configuration in Paraview~~
    - ~~If you can't, need to do post-processing in C++ and just display as vectors~~
  - Display 3D liquid crystal configurations in ParaView
* Read Convex Splitting paper by Cody
* ~~Read Selinger paper on rotating defect~~
* ~~Figure out how to apply a function to a finite element configuration in dealii~~
* ~~Make `LagrangeMultiplier` class a template with `order`~~
* ~~Update LagrangeMultiplier class to be useful in dealii~~
  - ~~Write function to return lagrange multiplier vector~~
  - ~~Write function to return Jacobian~~
* ~~Assert that Lagrange Multiplier errors are low enough, otherwise abort~~
* ~~Play around with making Lagrange Multiplier errors lower~~
* ~~Debug solver~~
  - ~~Try uniform configuration~~
  - ~~Confer with Cody about form of Newton's method~~
  - ~~Make sure boundary conditions are being applied correctly~~
  - ~~Write Laplace solver to make sure UMFPack is working properly~~
    - ~~Write with Dirichlet conditions~~
* ~~Learn how to use CMake~~
  - ~~Refactor all CMake files~~
* ~~Structure data~~
  - ~~Figure out how to make scripts/simulations agnostic to data location~~
  - ~~Structure data in a reasonably logical way~~