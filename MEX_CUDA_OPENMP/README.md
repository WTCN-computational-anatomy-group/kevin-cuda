# CUDA Cusparse Matlab

 * The CUDA-Cusparse code can handle/work with  any type of the input mxArrays, GPUarray or standard matlab CPU array as input   {prhs[0],prhs[1],prhs[2]  := mxGPUArray or CPU Array}[double or complex double]
 * Create sparse matrix  
 * Z=CuMatlab_sparse(X) 
 * Z=CuMatlab_sparse(X,Y)
 * Z=CuMatlab_sparse(X,Y,Z)
 * Z=CuMatlab_sparse(X,Y,Z,row,column) 
 * Z=CuMatlab_sparse(X,Y,Z,row,column,nz)
 * etc
 
 # CUDA Dense matrix manipulation Matlab
 
 * This CUDA code can handle/work with  any type of the input mxArrays,  GPUarray or standard matlab CPU array as input {prhs[0], prhs[1] := mxGPUArray or CPU Array}
 * gpuArray output, C=MM3D_CUBLAS(A,B,alpha) C=A*B*alpha.
 * etc

 
