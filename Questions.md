# Questions
------------------------
* Is it worth learning the convex splitting method?
  - If so, where do I look for that?
  - If not, what is the standard solver to use?
* How do I check whether I have the right solution?
  - Do I need a 2D configuration (i.e. a single defect with Dirichlet conditions)?
  - If so, I'll need to add that to the LagrangeMultiplier inversion scheme.
* How do you choose your degrees of freedom?
  - Why choose $\eta$ in that particular way?
  - Can you choose them in such a way to make the Jacobian symmetric and positive-definite?
* How do I check for symmetry and positive-definiteness?
* Where's a good place to look to learn ParaView?

* Should do 3D for now, can have MPI do slices.
* Don't worry about getting a matrix in a particular form
* Should learn convex splitting method -- might work in hydrodynamics
   - Helps Newton-Rhapson converge faster
* Trial and error for solver -- need one which doesn't require a particular form
* Check one defect -- line defect slice will look like a 2D thin film of liquid crystals
