# To Do
----------------
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
  - Compute the residual
  - Figure out how to set step size
  - Ouput results (have this in the other file)
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
* Figure out .vtu workflow for Matlab stuff
* ~~Read Selinger paper on rotating defect~~
* ~~Figure out how to apply a function to a finite element configuration in dealii~~
* ~~Make `LagrangeMultiplier` class a template with `order`~~
* Organize Code
  - Put classes which generate nematic configurations in their own files
  - Put classes which output director field in their own files
  - Figure out how to organize them -- maybe a big include file?
  - Could make one class which is just a template (which has different instantiations based on type)
  - Put boundary condition functions in separate file
* ~~Update LagrangeMultiplier class to be useful in dealii~~
  - ~~Write function to return lagrange multiplier vector~~
  - ~~Write function to return Jacobian~~
* Assert that Lagrange Multiplier errors are low enough, otherwise abort
* Play around with making Lagrange Multiplier errors lower
<span style="color:Gray">
* ~~Debug solver~~
  - ~~Try uniform configuration~~
  - ~~Confer with Cody about form of Newton's method~~
  - ~~Make sure boundary conditions are being applied correctly~~
  - Write Laplace solver to make sure UMFPack is working properly
    - ~~Write with Dirichlet conditions~~
    - Write with Neumann conditions
  - Try Neumann boundary conditions with nematic
  - Try LdG free energy with nematic
  - Redefine degrees of freedom to matcch with Cody's
  - Try with just eta, mu, nu
</span>
