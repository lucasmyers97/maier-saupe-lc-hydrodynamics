var lu__inversion__test_8cu =
[
    [ "BOOST_TEST_DYN_LINK", "lu__inversion__test_8cu.html#a139f00d2466d591f60b8d6a73c8273f1", null ],
    [ "cudaFree", "lu__inversion__test_8cu.html#a92d68e34be352574d2aff114d4493bae", null ],
    [ "cudaFree", "lu__inversion__test_8cu.html#af0c2fd0b116b8e8bef64cd2360a861a6", null ],
    [ "for", "lu__inversion__test_8cu.html#a8c38fab0e4846bc3f264442e7e7bf8f1", null ],
    [ "i< N;++i) for(unsigned int j=0;j< N;++j) lu_mat[n](i, j)=rand()/double(RAND_MAX);double *b=new double[N *num_mats];for(unsigned int n=0;n< num_mats;++n) { for(unsigned int i=0;i< N;++i) { b[N *n+i]=rand()/double(RAND_MAX);} } LUMatrixGPU< double, N > *d_lu_mat;double *d_b;cudaMalloc(&d_lu_mat, num_mats *sizeof(LUMatrixGPU< double, N >));cudaMalloc(&d_b, num_mats *N *sizeof(double));cudaMemcpy(d_lu_mat, lu_mat, num_mats *sizeof(LUMatrixGPU< double, N >), cudaMemcpyHostToDevice);cudaMemcpy(d_b, b, num_mats *N *sizeof(double), cudaMemcpyHostToDevice);matrix_inverse<<< 1, num_mats, num_mats *sizeof(LUMatrixGPU< double, N >)> >", "lu__inversion__test_8cu.html#ac0145b5facca671a62bd8e159fc3833e", null ],
    [ "matrix_inverse", "lu__inversion__test_8cu.html#a4fec6de571432d698b51ec460ae6849f", null ],
    [ "b", "lu__inversion__test_8cu.html#a2950103b8089ff16a46919fa1d1ece53", null ],
    [ "entry", "lu__inversion__test_8cu.html#aa751632eb4c6af6f1a19dde1ff4bed72", null ],
    [ "error", "lu__inversion__test_8cu.html#a2bf00f52622132f94d971b3baac4920c", null ],
    [ "lu_mat", "lu__inversion__test_8cu.html#a92a95fe9253c793b4932661527c40678", null ],
    [ "num_mats", "lu__inversion__test_8cu.html#a7301ee9b41123590b99ba1d02de44936", null ],
    [ "out_b", "lu__inversion__test_8cu.html#a728db0f9ff49efb62b3f02f7890764ac", null ]
];