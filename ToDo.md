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
  - Rewrite LagrangeMultiplier as a dealii Function class
  - Populate matrix
  - Find matrix solver appropriate for the problem
  - Figure out how to set step size
* Learn to use ParaView
  - ~~Display 2D liquid crystal configurations in ParaView~~
    - ~~Create function which returns Q-tensor for uniaxial configuration~~
    - ~~Project that function onto the finite element space~~
    - ~~Write DataPostprocessor function that gives the nematic configuration~~
    - ~~Write the nematic configuration to a .vtu file~~
    - ~~Open it in Paraview~~
    - See if you can make the nematic configuration in Paraview
    - If you can't, need to do post-processing in C++ and just display as vectors
  - Display 3D liquid crystal configurations in ParaView
* Read Convex Splitting paper by Cody
* Figure out .vtu workflow for Matlab stuff
* Read Selinger paper on rotating defect
* ~~Figure out how to apply a function to a finite element configuration in dealii~~
* ~~Make `LagrangeMultiplier` class a template with `order`~~
