# Algorithm overview:
---------------------
* As a first pass, each Q to be inverted will be done on a single thread
* All of the data needed for each inversion will be held in one object
* In shared memory on each block, there will be an array of these objects
1. Read Q's into global device memory
2. Create space for Lambdas and Jacs in global device memory
3. Run kernel, inputting the appropriate amount of shared memory
  - Read in the Q's from global device memory
  - Initialize Residual, Jacobian, and Lambda values on shared memory
  - Solve linear equation with solver
  - Add solution to current Lambda value
  - Calculate integrals from Lambda via Lebedev quadrature
  - Assemble Residual and Jacobian from integrals
  - Check residual value
  - If it's good, write to global memory
  - If not, go back to "Solve linear equation" step
4. Read from global device memory to get Lambda values and Jacobians
5. Repeat in batches until all Q values are inverted
