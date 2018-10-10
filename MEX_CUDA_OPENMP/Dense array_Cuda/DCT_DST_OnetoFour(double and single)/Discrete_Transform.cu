
/*
 * Discrete Cosine/Sine Transform(DCT/DST and IDCT/IDST one to four-all in one)
 * DCT/DST and IDCT/IDST I ---> IV
 * This CUDA code can handle/work with  any type of the input mxArrays, 
 * GPUarray or standard matlab CPU array as input {prhs[0] := mxGPUArray or CPU Array}
 * GpuArray/cpuArray output, B=Discrete_Transform(A, , type of Transform (sine or cosine), type of Transform(direct/inverse), type of DCT/DST or IDCT/IDST, dimensions).
 * Developed at UCL, Institute of Neurology, 12 Queen Square, WC1N 3AR, London
 * Wellcome Trust Centre for Neuroimaging
 * Part of the project SPM(http://www.fil.ion.ucl.ac.uk/spm)
 * Copyright 2018
 * Kevin Bronik
 */

#include "matrix.h"
#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include "Discrete_TransformD.cu"
#include "Discrete_TransformS.cu"

extern "C" static void mexTransD(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[]);
                 
extern "C" static void mexTransS(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[]);
                 

                 
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
 int nDevices;
cudaError_t errCode =cudaGetDeviceCount(&nDevices); 
//int nDevices;
//cudaGetDeviceCount(&nDevices);

if (errCode != cudaSuccess){
printf("Error! No CUDA devices found! \n");
return;
}

    
   if (nrhs==5 && nlhs==1) {

        if (mxIsGPUArray(prhs[0])) {
        
    mxGPUArray const *tempGPU;
    tempGPU = mxGPUCreateFromMxArray(prhs[0]);
        
       if (mxGPUGetClassID(tempGPU) == mxDOUBLE_CLASS && mxGPUGetComplexity(tempGPU) == mxREAL){ 
           
        mexTransD(nlhs, plhs,
                 nrhs, prhs);
           
       mxGPUDestroyGPUArray(tempGPU);
       return;
           }
       else if (mxGPUGetClassID(tempGPU) == mxSINGLE_CLASS && mxGPUGetComplexity(tempGPU) == mxREAL){ 
        
		mexTransS(nlhs, plhs,
                 nrhs, prhs);
       
       mxGPUDestroyGPUArray(tempGPU);
       return;
           
           }
           
       else{ 
               
       mxGPUDestroyGPUArray(tempGPU);
       mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Incorrect input arguments! %s\n");    
           
           }
    }
    //
     else if(!mxIsGPUArray(prhs[0])) {  
       if (mxGetClassID(prhs[0]) == mxDOUBLE_CLASS && (!mxIsComplex(prhs[0]))){ 
            
        mexTransD(nlhs, plhs,
                 nrhs, prhs);
                 return; 
       
           }
       else if (mxGetClassID(prhs[0]) == mxSINGLE_CLASS && (!mxIsComplex(prhs[0]))){ 
           
        mexTransS(nlhs, plhs,
                 nrhs, prhs);
                 return; 
           }
           
           
       else{  
               
       mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Incorrect input arguments! %s\n");    
           
           }
           
       }           
                 
    }
    
 
    
     else if  ((nrhs<5) || (nrhs>5)  ||  (nlhs<1)  || (nlhs>1) ) {
            
        mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Incorrect input/output arguments! input argument must be five and output arguments must be one\n"); 
                return;    
        }

}