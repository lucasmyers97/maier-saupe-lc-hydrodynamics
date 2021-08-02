# Algorithm overview:
---------------------
* As a first pass, each Q to be inverted will be done on a block.
* Each block will consit of one warp (32 threads).
1. First compute all `n_lebedev_pts` of $\exp(\Lambda_{kl} \xi_k \xi_l)$ with thread striding -- store on shared memory
2. Assign each thread to an integral to compute -- store the integral results on shared memory
3. Have each thread compute entry in Jacobian or Residual using integrals, write to global memory
4. Solve matrix inversion problem with cuBLAS batched inversion
5. Check 
