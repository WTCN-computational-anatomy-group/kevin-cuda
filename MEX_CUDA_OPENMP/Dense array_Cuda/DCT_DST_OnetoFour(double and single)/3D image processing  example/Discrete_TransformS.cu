
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
#include "CuFilesS/Discrete_Transforms_kernel.cuh"
#include "CuFilesS/DCT_I_Column.cu"
#include "CuFilesS/DCT_I_Row.cu"
#include "CuFilesS/DCT_I_Column_Inverse.cu"
#include "CuFilesS/DCT_I_Row_Inverse.cu"
#include "CuFilesS/DCT_II_Row.cu"
#include "CuFilesS/DCT_II_Row_Inverse.cu"
#include "CuFilesS/DCT_II_Column.cu"
#include "CuFilesS/DCT_II_Column_Inverse.cu"
#include "CuFilesS/DCT_III_Row.cu"
#include "CuFilesS/DCT_III_Row_Inverse.cu"
#include "CuFilesS/DCT_III_Column.cu"
#include "CuFilesS/DCT_III_Column_Inverse.cu"
#include "CuFilesS/DCT_IV_Row.cu"
#include "CuFilesS/DCT_IV_Row_Inverse.cu"
#include "CuFilesS/DCT_IV_Column.cu"
#include "CuFilesS/DCT_IV_Column_Inverse.cu"
#include "CuFilesS/DST_I_Column.cu"
#include "CuFilesS/DST_I_Row.cu"
#include "CuFilesS/DST_I_Column_Inverse.cu"
#include "CuFilesS/DST_I_Row_Inverse.cu"
#include "CuFilesS/DST_II_Row.cu"
#include "CuFilesS/DST_II_Row_Inverse.cu"
#include "CuFilesS/DST_II_Column.cu"
#include "CuFilesS/DST_II_Column_Inverse.cu"
#include "CuFilesS/DST_III_Row.cu"
#include "CuFilesS/DST_III_Row_Inverse.cu"
#include "CuFilesS/DST_III_Column.cu"
#include "CuFilesS/DST_III_Column_Inverse.cu"
#include "CuFilesS/DST_IV_Row.cu"
#include "CuFilesS/DST_IV_Row_Inverse.cu"
#include "CuFilesS/DST_IV_Column.cu"
#include "CuFilesS/DST_IV_Column_Inverse.cu"
//#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#define DEFAULT_DIM 32   
#define 	DELTA(i, j)   ((i==j)?1:0)

// DCT
extern "C" void  CalculateTransformDCTColumnOneS(float * A, float * C, int numARows,
	int numAColumns, int numCRows, int numCColumns);
    
extern "C" void  CalculateTransformDCTInverseColumnOneS(float * A, float * C, int numARows,
	int numAColumns, int numCRows, int numCColumns);
    
extern "C" void  CalculateTransformDCTRowOneS(float * A, float * C, int numARows,
	int numAColumns, int numCRows, int numCColumns);
    
extern "C" void  CalculateTransformDCTInverseRowOneS(float * A, float * C, int numARows,
	int numAColumns, int numCRows, int numCColumns);    
    
extern "C" void  CalculateTransformDCTRowTwoS(float * A, float * C, int numARows,
	int numAColumns, int numCRows, int numCColumns);
    
extern "C" void  CalculateTransformDCTInverseRowTwoS(float * A, float * C, int numARows,
	int numAColumns, int numCRows, int numCColumns);
    
extern "C" void  CalculateTransformDCTColumnTwoS(float * A, float * C, int numARows,
	int numAColumns, int numCRows, int numCColumns);
    
extern "C" void  CalculateTransformDCTInverseColumnTwoS(float * A, float * C, int numARows,
	int numAColumns, int numCRows, int numCColumns);
    
extern "C" void  CalculateTransformDCTColumnThreeS(float * A, float * C, int numARows,
	int numAColumns, int numCRows, int numCColumns); 
    
extern "C" void  CalculateTransformDCTInverseColumnThreeS(float * A, float * C, int numARows,
	int numAColumns, int numCRows, int numCColumns);
    
extern "C" void  CalculateTransformDCTRowThreeS(float * A, float * C, int numARows,
	int numAColumns, int numCRows, int numCColumns);
    
extern "C" void  CalculateTransformDCTInverseRowThreeS(float * A, float * C, int numARows,
	int numAColumns, int numCRows, int numCColumns);
    
extern "C" void  CalculateTransformDCTColumnFourS(float * A, float * C, int numARows,
	int numAColumns, int numCRows, int numCColumns);  
    
extern "C" void  CalculateTransformDCTInverseColumnFourS(float * A, float * C, int numARows,
	int numAColumns, int numCRows, int numCColumns);
    
extern "C" void  CalculateTransformDCTRowFourS(float * A, float * C, int numARows,
	int numAColumns, int numCRows, int numCColumns);
    
extern "C" void  CalculateTransformDCTInverseRowFourS(float * A, float * C, int numARows,
	int numAColumns, int numCRows, int numCColumns); 
    
    // DST
extern "C" void  CalculateTransformDSTColumnOneS(float * A, float * C, int numARows,
	int numAColumns, int numCRows, int numCColumns);
    
extern "C" void  CalculateTransformDSTInverseColumnOneS(float * A, float * C, int numARows,
	int numAColumns, int numCRows, int numCColumns);
    
extern "C" void  CalculateTransformDSTRowOneS(float * A, float * C, int numARows,
	int numAColumns, int numCRows, int numCColumns);
    
extern "C" void  CalculateTransformDSTInverseRowOneS(float * A, float * C, int numARows,
	int numAColumns, int numCRows, int numCColumns);    
    
extern "C" void  CalculateTransformDSTRowTwoS(float * A, float * C, int numARows,
	int numAColumns, int numCRows, int numCColumns);
    
extern "C" void  CalculateTransformDSTInverseRowTwoS(float * A, float * C, int numARows,
	int numAColumns, int numCRows, int numCColumns);
    
extern "C" void  CalculateTransformDSTColumnTwoS(float * A, float * C, int numARows,
	int numAColumns, int numCRows, int numCColumns);
    
extern "C" void  CalculateTransformDSTInverseColumnTwoS(float * A, float * C, int numARows,
	int numAColumns, int numCRows, int numCColumns);
    
extern "C" void  CalculateTransformDSTColumnThreeS(float * A, float * C, int numARows,
	int numAColumns, int numCRows, int numCColumns); 
    
extern "C" void  CalculateTransformDSTInverseColumnThreeS(float * A, float * C, int numARows,
	int numAColumns, int numCRows, int numCColumns);
    
extern "C" void  CalculateTransformDSTRowThreeS(float * A, float * C, int numARows,
	int numAColumns, int numCRows, int numCColumns);
    
extern "C" void  CalculateTransformDSTInverseRowThreeS(float * A, float * C, int numARows,
	int numAColumns, int numCRows, int numCColumns);
    
extern "C" void  CalculateTransformDSTColumnFourS(float * A, float * C, int numARows,
	int numAColumns, int numCRows, int numCColumns);  
    
extern "C" void  CalculateTransformDSTInverseColumnFourS(float * A, float * C, int numARows,
	int numAColumns, int numCRows, int numCColumns);
    
extern "C" void  CalculateTransformDSTRowFourS(float * A, float * C, int numARows,
	int numAColumns, int numCRows, int numCColumns);
    
extern "C" void  CalculateTransformDSTInverseRowFourS(float * A, float * C, int numARows,
	int numAColumns, int numCRows, int numCColumns); 
    
extern "C" static void mexTransS(int nlhs, mxArray *plhs[],
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

char row[] = "row";
char column[] = "column";
char one[] = "one";
char two[] = "two";
char three[] = "three";
char four[] = "four";

char direct[] = "direct";
char inverse[] = "inverse";
char cosine[] = "cosine";
char sine[] = "sine";
 
    char const * const InputErrMsg = "Invalid input to MEX file, input(FIRST ARGUMENT) must be single precision (float), and the number of input arguments must be five.";
    
    if ((nrhs!=5)) {
        mexErrMsgIdAndTxt("MATLAB:mexatexit:invalidInput", InputErrMsg);
    }

char *input_buf0;
 input_buf0 = mxArrayToString(prhs[0]);
 char *input_buf1;
 input_buf1 = mxArrayToString(prhs[1]);
char *input_buf2;
 input_buf2 = mxArrayToString(prhs[2]);
  char *input_buf3;
 input_buf3 = mxArrayToString(prhs[3]);
char *input_buf4;
 input_buf4 = mxArrayToString(prhs[4]);
    if ((mxIsChar(prhs[0]))){
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Input(FIRST ARGUMENT) must be array, or gpuArray object not  %s\n",input_buf0);
    }
     if (!(mxIsChar(prhs[1]))){
        mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Input(SECOND ARGUMENT) must be of type string.\n.");
    }
      if (!(mxIsChar(prhs[2]))){
        mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Input(THIRD ARGUMENT) must be of type string.\n.");
    }
        if (!(mxIsChar(prhs[3]))){
        mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Input(FOURTH ARGUMENT) must be of type string.\n.");
    }
      if (!(mxIsChar(prhs[4]))){
        mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Input(FIFTH ARGUMENT) must be of type string.\n.");
    } 
  
  if ((strcmp (cosine,input_buf1) != 0) &&(strcmp (sine,input_buf1) != 0) )
{
    mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Input(SECOND ARGUMENT) must be 'cosine' or 'sine'  not  %s\n",input_buf1);
   }
    
  if ((strcmp (direct,input_buf2) != 0)&& (strcmp (inverse,input_buf2) != 0) )
{
        mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Input(THIRD ARGUMENT) must be 'direct' or 'inverse' not  %s\n",input_buf2);
    }
    

   if ((strcmp (one,input_buf3) != 0)&& (strcmp (two,input_buf3) != 0) && (strcmp (three,input_buf3) != 0) && (strcmp (four,input_buf3) != 0))
{
                mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Input(FOURTH ARGUMENT) must be 'one' or 'two' or 'three' or 'four' not  %s\n",input_buf3);

    }
       if ((strcmp (column,input_buf4) != 0)&&(strcmp (row,input_buf4) != 0))
{  
            mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Input(FIFTH ARGUMENT) must be 'column' or 'row' not  %s\n",input_buf4);
    }
 //COSINE TRANSFORM   
 if (strcmp (cosine,input_buf1) == 0)
{

 if (strcmp (direct,input_buf2) == 0)
{   
  if (strcmp (column,input_buf4) == 0)
{   
 
    if (mxIsGPUArray(prhs[0])) {
    
  

    mxGPUArray const *A;
    mxGPUArray *B;
    float const *d_A;
    float *d_B;
    int numARows, numAColumns, numCRows,  numCColumns;
    mxInitGPU();
    cudaError_t error;
    int devID = 0;
    error = cudaGetDevice(&devID);
    cudaDeviceProp deviceProp;
     error = cudaGetDeviceProperties(&deviceProp, devID);
      if (error != cudaSuccess)
      {
          printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);
          exit(EXIT_FAILURE);
      }
    int TILEDIM = (deviceProp.major < 2) ? 16 : 32;

   A = mxGPUCreateFromMxArray(prhs[0]);
   if(mxGPUGetComplexity(A) != mxREAL){
       mxGPUDestroyGPUArray(A);
       mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Incorrect input arguments!, input matrix must be real %s\n");
       
       }
   
   const mwSize *dims;
   dims=mxGPUGetDimensions(A);
   numARows = (int)dims[0]; /* gets number of rows of A */
   numAColumns = (int)dims[1]; /* gets number of columns of A */
   size_t pivot_dimensA[2] = {numARows,numAColumns};
   mwSize NrOfDim=mxGPUGetNumberOfDimensions(A);
   
		numCRows = numARows;
		numCColumns = numAColumns;
  if (numARows==1)
 {   
 printf("Attention, this is a row vector, please try Discrete Cosine Transform in row wise \n");
 return;
 }
 
    char const * const errId = "parallel:gpu:mexGPUExample:InvalidInput";
    char const * const errMsg = "Invalid input to MEX file, input(FIRST ARGUMENT) must be single precision (float).";

    if (mxGPUGetClassID(A) != mxSINGLE_CLASS) {
        mexErrMsgIdAndTxt(errId, errMsg);
    }

    d_A = (float const *)(mxGPUGetDataReadOnly(A));
    mxGPUDestroyGPUArray(A);
    B = mxGPUCreateGPUArray(NrOfDim, (mwSize*) pivot_dimensA, mxSINGLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    d_B = (float *)(mxGPUGetData(B));
    dim3 dimBlock;
	dim3 dimGrid;
    
     switch (TILEDIM){
        
        case 16:
            
     TILE_DIM= TILEDIM;
	 dimBlock.x=TILE_DIM;
	 dimBlock.y=TILE_DIM;
     dimBlock.z=1;
	dimGrid.x = (numCColumns + dimBlock.x - 1) / dimBlock.x;
	dimGrid.y = (numCRows + dimBlock.y - 1) / dimBlock.y;
 if (strcmp (one,input_buf3) == 0)
{
     
   DCTI_Column_Kernel_GPUAS <16> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
   
   
 }  
 if (strcmp (two,input_buf3) == 0)
{
     
   DCTII_Column_Kernel_GPUAS <16> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
   
   
 }
 if (strcmp (three,input_buf3) == 0)
{
     
   DCTIII_Column_Kernel_GPUAS <16> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
   
   
 }
 if (strcmp (four,input_buf3) == 0)
{
     
   DCTIV_Column_Kernel_GPUAS <16> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
   
   
 }

    plhs[0] = mxGPUCreateMxArrayOnGPU(B);   
    mxGPUDestroyGPUArray(B);
    
         case 32:
            
     TILE_DIM= TILEDIM;
	 dimBlock.x=TILE_DIM;
	 dimBlock.y=TILE_DIM;
     dimBlock.z=1;
	dimGrid.x = (numCColumns + dimBlock.x - 1) / dimBlock.x;
	dimGrid.y = (numCRows + dimBlock.y - 1) / dimBlock.y;
    
     if (strcmp (one,input_buf3) == 0)
{
     
   DCTI_Column_Kernel_GPUAS <32> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
   
   
 }  
 if (strcmp (two,input_buf3) == 0)
{
     
   DCTII_Column_Kernel_GPUAS <32> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
   
   
 }
 if (strcmp (three,input_buf3) == 0)
{
     
   DCTIII_Column_Kernel_GPUAS <32> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
   
   
 }
 if (strcmp (four,input_buf3) == 0)
{
     
   DCTIV_Column_Kernel_GPUAS <32> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
   
   
 }

    plhs[0] = mxGPUCreateMxArrayOnGPU(B);  
    mxGPUDestroyGPUArray(B);
    
      } 
    }

    else if (!(mxIsGPUArray(prhs[0]))){
             
   if (mxGetClassID(prhs[0]) != mxSINGLE_CLASS) {
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Invalid input to MEX file, input(FIRST ARGUMENT) must be single precision (float).");
    } 
   if(mxIsComplex(prhs[0])){
       
       mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Incorrect input arguments!, input matrix must be real %s\n");
       
       }
    
  	int numARows = (int)mxGetM(prhs[0]); 		// number of rows in the matrix A
	int numAColumns = (int)mxGetN(prhs[0]); 	// number of columns in the matrix A
	int numCRows;		// number of rows in the matrix C (you have to set this)
	int numCColumns;	// number of columns in the matrix C (you have to set this)
    	
		numCRows = numARows;
		numCColumns = numAColumns;
        
 if (numARows==1)
 {   
 printf("Attention, this is a row vector, please try Discrete Cosine Transform in row wise \n");
 return;
 }
    mxInitGPU();
	float  * hostA ; // The A matrix
	hostA = (float *)mxGetData(prhs[0]);   
    plhs[0] = mxCreateNumericMatrix(numCRows, numCColumns, mxSINGLE_CLASS, mxREAL);
    float  *pointer =(float*) mxGetPr(plhs[0]);
  
     if (strcmp (one,input_buf3) == 0)
    {
     
       CalculateTransformDCTColumnOneS(hostA, pointer, numARows, numAColumns, numCRows, numCColumns);
	
    }
     if (strcmp (two,input_buf3) == 0)
    {
     
       CalculateTransformDCTColumnTwoS(hostA, pointer, numARows, numAColumns, numCRows, numCColumns);
	
    }
     if (strcmp (three,input_buf3) == 0)
    {
     
       CalculateTransformDCTColumnThreeS(hostA, pointer, numARows, numAColumns, numCRows, numCColumns);
	
    }
     if (strcmp (four,input_buf3) == 0)
    {
     
       CalculateTransformDCTColumnFourS(hostA, pointer, numARows, numAColumns, numCRows, numCColumns);
	
         }	
       } 
    } 
   
     if (strcmp (row,input_buf4) == 0)
{

    if (mxIsGPUArray(prhs[0])) {
		
    mxGPUArray const *A;
    mxGPUArray *B;
    float const *d_A;
    float *d_B;
    int numARows, numAColumns, numCRows,  numCColumns;
    mxInitGPU();
    cudaError_t error;
    int devID = 0;
    error = cudaGetDevice(&devID);
    cudaDeviceProp deviceProp;
     error = cudaGetDeviceProperties(&deviceProp, devID);
      if (error != cudaSuccess)
      {
          printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);
          exit(EXIT_FAILURE);
      }
    int TILEDIM = (deviceProp.major < 2) ? 16 : 32;
    A = mxGPUCreateFromMxArray(prhs[0]);
       if(mxGPUGetComplexity(A) != mxREAL){
       mxGPUDestroyGPUArray(A);
       mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Incorrect input arguments!, input matrix must be real %s\n");
       
       }
    const mwSize *dims;
    dims=mxGPUGetDimensions(A);
    numARows = (int)dims[0]; /* gets number of rows of A */
    numAColumns = (int)dims[1]; /* gets number of columns of A */
   size_t pivot_dimensA[2] = {numARows,numAColumns};
   mwSize NrOfDim=mxGPUGetNumberOfDimensions(A); 
 
 if (numAColumns==1)
 {   
 printf("Attention, this is a column vector, please try Discrete Cosine Transform in column wise \n");
 return;
 }
 
    numCRows = numARows;
	numCColumns = numAColumns;

   char const * const errId = "parallel:gpu:mexGPUExample:InvalidInput";
   char const * const errMsg = "Invalid input to MEX file, input(FIRST ARGUMENT) must be single precision (float).";
    if (mxGPUGetClassID(A) != mxSINGLE_CLASS) {
        mexErrMsgIdAndTxt(errId, errMsg);
    }
    
    d_A = (float const *)(mxGPUGetDataReadOnly(A));
    mxGPUDestroyGPUArray(A);
    B = mxGPUCreateGPUArray(NrOfDim, (mwSize*) pivot_dimensA, mxSINGLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    d_B = (float *)(mxGPUGetData(B));
    dim3 dimBlock;
	dim3 dimGrid;
    
     switch (TILEDIM){
        
        case 16:
            
     TILE_DIM= TILEDIM;
	 dimBlock.x=TILE_DIM;
	 dimBlock.y=TILE_DIM;
     dimBlock.z=1;
	dimGrid.x = (numCColumns + dimBlock.x - 1) / dimBlock.x;
	dimGrid.y = (numCRows + dimBlock.y - 1) / dimBlock.y;
 if (strcmp (one,input_buf3) == 0)
{
    
   DCTI_Row_Kernel_GPUAS <16> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
}
 if (strcmp (two,input_buf3) == 0)
{
    
   DCTII_Row_Kernel_GPUAS <16> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
}
 if (strcmp (three,input_buf3) == 0)
{
    
   DCTIII_Row_Kernel_GPUAS <16> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
}
 if (strcmp (four,input_buf3) == 0)
{
    
   DCTIV_Row_Kernel_GPUAS <16> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
}
    
    plhs[0] = mxGPUCreateMxArrayOnGPU(B); 
    mxGPUDestroyGPUArray(B);
    
         case 32:
            
     TILE_DIM= TILEDIM;
	 dimBlock.x=TILE_DIM;
	 dimBlock.y=TILE_DIM;
     dimBlock.z=1;
	dimGrid.x = (numCColumns + dimBlock.x - 1) / dimBlock.x;
	dimGrid.y = (numCRows + dimBlock.y - 1) / dimBlock.y;
    
     if (strcmp (one,input_buf3) == 0)
{
    
   DCTI_Row_Kernel_GPUAS <32> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
}
 if (strcmp (two,input_buf3) == 0)
{
    
   DCTII_Row_Kernel_GPUAS <32> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
}
 if (strcmp (three,input_buf3) == 0)
{
    
   DCTIII_Row_Kernel_GPUAS <32> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
}
 if (strcmp (four,input_buf3) == 0)
{
    
   DCTIV_Row_Kernel_GPUAS <32> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
}

    plhs[0] = mxGPUCreateMxArrayOnGPU(B);
    mxGPUDestroyGPUArray(B);
     
        }    
	}

    else if (!(mxIsGPUArray(prhs[0]))){
            
   if (mxGetClassID(prhs[0]) != mxSINGLE_CLASS) {
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Invalid input to MEX file, input(FIRST ARGUMENT) must be single precision (float).");
    }  
   if(mxIsComplex(prhs[0])){
       
       mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Incorrect input arguments!, input matrix must be real %s\n");
       
       }
  	int numARows = (int)mxGetM(prhs[0]); 		// number of rows in the matrix A
	int numAColumns = (int)mxGetN(prhs[0]); 	// number of columns in the matrix A
	
	int numCRows;		// number of rows in the matrix C (you have to set this)
	int numCColumns;	// number of columns in the matrix C (you have to set this)
	
	numCRows = numARows;
	numCColumns = numAColumns;
	float  * hostA ; // The A matrix
	
 if (numAColumns==1)
 {   
 printf("Attention, this is a column vector, please try Discrete Cosine Transform in column wise \n");
 return;
 }

    mxInitGPU();
	hostA = (float *)mxGetData(prhs[0]);
    plhs[0] = mxCreateNumericMatrix(numCRows, numCColumns, mxSINGLE_CLASS, mxREAL);
    float  *pointer = (float*)mxGetPr(plhs[0]);
     if (strcmp (one,input_buf3) == 0)
{  
      CalculateTransformDCTRowOneS(hostA, pointer, numARows, numAColumns, numCRows, numCColumns);
      
      } 
       if (strcmp (two,input_buf3) == 0)
{  
      CalculateTransformDCTRowTwoS(hostA, pointer, numARows, numAColumns, numCRows, numCColumns);
      
      }
       if (strcmp (three,input_buf3) == 0)
{  
      CalculateTransformDCTRowThreeS(hostA, pointer, numARows, numAColumns, numCRows, numCColumns);
      
      }
       if (strcmp (four,input_buf3) == 0)
{  
      CalculateTransformDCTRowFourS(hostA, pointer, numARows, numAColumns, numCRows, numCColumns);
      
        }
  
      } 

   }

}

 if (strcmp (inverse,input_buf2) == 0)
{
    if (strcmp (column,input_buf4) == 0)
{      
 
    if (mxIsGPUArray(prhs[0])) {

    mxGPUArray const *A;
    mxGPUArray *B;
    float const *d_A;
    float *d_B;
    int numARows, numAColumns, numCRows,  numCColumns;
    mxInitGPU();
    cudaError_t error;
    int devID = 0;
    error = cudaGetDevice(&devID);
    cudaDeviceProp deviceProp;
     error = cudaGetDeviceProperties(&deviceProp, devID);
      if (error != cudaSuccess)
      {
          printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);
          exit(EXIT_FAILURE);
      }
    int TILEDIM = (deviceProp.major < 2) ? 16 : 32;
    A = mxGPUCreateFromMxArray(prhs[0]);
       if(mxGPUGetComplexity(A) != mxREAL){
       mxGPUDestroyGPUArray(A);
       mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Incorrect input arguments!, input matrix must be real %s\n");
       
       }
    const mwSize *dims;
   dims=mxGPUGetDimensions(A);
   numARows = (int)dims[0]; /* gets number of rows of A */
   numAColumns = (int)dims[1]; /* gets number of columns of A */
   size_t pivot_dimensA[2] = {numARows,numAColumns};
   mwSize NrOfDim=mxGPUGetNumberOfDimensions(A);
   
		numCRows = numARows;
		numCColumns = numAColumns;
  if (numARows==1)
 {   
 printf("Attention, this is a row vector, please try Inverse Discrete Cosine Transform in row wise \n");
 return;
 }
 
    char const * const errId = "parallel:gpu:mexGPUExample:InvalidInput";
    char const * const errMsg = "Invalid input to MEX file, input(FIRST ARGUMENT) must be single precision (float).";

    if (mxGPUGetClassID(A) != mxSINGLE_CLASS) {
        mexErrMsgIdAndTxt(errId, errMsg);
    }

    d_A = (float const *)(mxGPUGetDataReadOnly(A));
    mxGPUDestroyGPUArray(A);
    B = mxGPUCreateGPUArray(NrOfDim, (mwSize*) pivot_dimensA, mxSINGLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    d_B = (float *)(mxGPUGetData(B));
    dim3 dimBlock;
	dim3 dimGrid;
    
     switch (TILEDIM){
        
        case 16:
            
     TILE_DIM= TILEDIM;
	 dimBlock.x=TILE_DIM;
	 dimBlock.y=TILE_DIM;
     dimBlock.z=1;
	dimGrid.x = (numCColumns + dimBlock.x - 1) / dimBlock.x;
	dimGrid.y = (numCRows + dimBlock.y - 1) / dimBlock.y;
 if (strcmp (one,input_buf3) == 0)
{
    
   DCTI_Column_Inverse_Kernel_GPUAS <16> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
   
   
}
  if (strcmp (two,input_buf3) == 0)
{
    
   DCTII_Column_Inverse_Kernel_GPUAS <16> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
   
   
}
 if (strcmp (three,input_buf3) == 0)
{
    
   DCTIII_Column_Inverse_Kernel_GPUAS <16> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
   
   
}
 if (strcmp (four,input_buf3) == 0)
{
    
   DCTIV_Column_Inverse_Kernel_GPUAS <16> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
   
   
}

    plhs[0] = mxGPUCreateMxArrayOnGPU(B);
    mxGPUDestroyGPUArray(B);
     
    case 32:
            
     TILE_DIM= TILEDIM;
	 dimBlock.x=TILE_DIM;
	 dimBlock.y=TILE_DIM;
     dimBlock.z=1;
	dimGrid.x = (numCColumns + dimBlock.x - 1) / dimBlock.x;
	dimGrid.y = (numCRows + dimBlock.y - 1) / dimBlock.y;
     if (strcmp (one,input_buf3) == 0)
{
    
   DCTI_Column_Inverse_Kernel_GPUAS <32> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
   
   
}
  if (strcmp (two,input_buf3) == 0)
{
    
   DCTII_Column_Inverse_Kernel_GPUAS <32> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
   
   
}
 if (strcmp (three,input_buf3) == 0)
{
    
   DCTIII_Column_Inverse_Kernel_GPUAS <32> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
   
   
}
 if (strcmp (four,input_buf3) == 0)
{
    
   DCTIV_Column_Inverse_Kernel_GPUAS <32> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
   
   
}
 
    plhs[0] = mxGPUCreateMxArrayOnGPU(B);   
    mxGPUDestroyGPUArray(B);
    
       }   
	}

    else if (!(mxIsGPUArray(prhs[0]))){
  
    if (mxGetClassID(prhs[0]) != mxSINGLE_CLASS) {
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Invalid input to MEX file, input(FIRST ARGUMENT) must be single precision (float).");
    }        
    if(mxIsComplex(prhs[0])){
       
       mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Incorrect input arguments!, input matrix must be real %s\n");
       
       }           
  	int numARows = (int)mxGetM(prhs[0]); 		// number of rows in the matrix A
	int numAColumns = (int)mxGetN(prhs[0]); 	// number of columns in the matrix A
	int numCRows;		// number of rows in the matrix C (you have to set this)
	int numCColumns;	// number of columns in the matrix C (you have to set this)
    	
		numCRows = numARows;
		numCColumns = numAColumns;
        
 if (numARows==1)
 {   
 printf("Attention, this is a row vector, please try Inverse Discrete Cosine Transform in row wise \n");
 return;
 }
    mxInitGPU();

	float  * hostA ; // The A matrix
	hostA = (float *)mxGetData(prhs[0]);
    plhs[0] = mxCreateNumericMatrix(numCRows, numCColumns, mxSINGLE_CLASS, mxREAL);
    float  *pointer = (float*)mxGetPr(plhs[0]);
     
     if (strcmp (one,input_buf3) == 0)
   {
  
       CalculateTransformDCTInverseColumnOneS(hostA, pointer, numARows, numAColumns, numCRows, numCColumns);
	
   }
     if (strcmp (two,input_buf3) == 0)
   {
  
       CalculateTransformDCTInverseColumnTwoS(hostA, pointer, numARows, numAColumns, numCRows, numCColumns);
	
   }
     if (strcmp (three,input_buf3) == 0)
   {
  
       CalculateTransformDCTInverseColumnThreeS(hostA, pointer, numARows, numAColumns, numCRows, numCColumns);
	
   }
     if (strcmp (four,input_buf3) == 0)
   {
  
       CalculateTransformDCTInverseColumnFourS(hostA, pointer, numARows, numAColumns, numCRows, numCColumns);
	
      }	
   } 
}

    if (strcmp (row,input_buf4) == 0)
{
     
    if (mxIsGPUArray(prhs[0])) {

    mxGPUArray const *A;
    mxGPUArray *B;
    float const *d_A;
   
    float *d_B;
    int numARows, numAColumns, numCRows,  numCColumns;
    mxInitGPU();
    cudaError_t error;
    int devID = 0;
    error = cudaGetDevice(&devID);
    cudaDeviceProp deviceProp;
     error = cudaGetDeviceProperties(&deviceProp, devID);
      if (error != cudaSuccess)
      {
          printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);
          exit(EXIT_FAILURE);
      }
    int TILEDIM = (deviceProp.major < 2) ? 16 : 32;
    
    A = mxGPUCreateFromMxArray(prhs[0]);
       if(mxGPUGetComplexity(A) != mxREAL){
       mxGPUDestroyGPUArray(A);
       mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Incorrect input arguments!, input matrix must be real %s\n");
       
       }
    const mwSize *dims;
    dims=mxGPUGetDimensions(A);
    numARows = (int)dims[0]; /* gets number of rows of A */
    numAColumns = (int)dims[1]; /* gets number of columns of A */
    size_t pivot_dimensA[2] = {numARows,numAColumns};
    mwSize NrOfDim=mxGPUGetNumberOfDimensions(A);
    
  if (numAColumns==1)
 {   
 printf("Attention, this is a column vector, please try Inverse Discrete Cosine Transform in column wise \n");
 return;
 }
 
    numCRows = numARows;
	numCColumns = numAColumns;
    char const * const errId = "parallel:gpu:mexGPUExample:InvalidInput";
    char const * const errMsg = "Invalid input to MEX file, input(FIRST ARGUMENT) must be single precision (float).";

    if (mxGPUGetClassID(A) != mxSINGLE_CLASS) {
        mexErrMsgIdAndTxt(errId, errMsg);
    }

    d_A = (float const *)(mxGPUGetDataReadOnly(A));
    mxGPUDestroyGPUArray(A);
    B = mxGPUCreateGPUArray(NrOfDim, (mwSize*) pivot_dimensA, mxSINGLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    d_B = (float *)(mxGPUGetData(B));
    dim3 dimBlock;
	dim3 dimGrid;
    
     switch (TILEDIM){
        
        case 16:
            
     TILE_DIM= TILEDIM;
	 dimBlock.x=TILE_DIM;
	 dimBlock.y=TILE_DIM;
     dimBlock.z=1;
	dimGrid.x = (numCColumns + dimBlock.x - 1) / dimBlock.x;
	dimGrid.y = (numCRows + dimBlock.y - 1) / dimBlock.y;
 if (strcmp (one,input_buf3) == 0)
{
   DCTI_Row__InverseKernel_GPUAS <16> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);

   }
 if (strcmp (two,input_buf3) == 0)
{
   DCTII_Row__InverseKernel_GPUAS <16> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);

   }
 if (strcmp (three,input_buf3) == 0)
{
   DCTIII_Row__InverseKernel_GPUAS <16> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);

   }
 if (strcmp (four,input_buf3) == 0)
{
   DCTIV_Row__InverseKernel_GPUAS <16> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);

   }

    plhs[0] = mxGPUCreateMxArrayOnGPU(B);   
    mxGPUDestroyGPUArray(B);
    
        case 32:
            
     TILE_DIM= TILEDIM;
	 dimBlock.x=TILE_DIM;
	 dimBlock.y=TILE_DIM;
     dimBlock.z=1;
	dimGrid.x = (numCColumns + dimBlock.x - 1) / dimBlock.x;
	dimGrid.y = (numCRows + dimBlock.y - 1) / dimBlock.y;
    
     if (strcmp (one,input_buf3) == 0)
{
   DCTI_Row__InverseKernel_GPUAS <32> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);

   }
 if (strcmp (two,input_buf3) == 0)
{
   DCTII_Row__InverseKernel_GPUAS <32> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);

   }
 if (strcmp (three,input_buf3) == 0)
{
   DCTIII_Row__InverseKernel_GPUAS <32> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);

   }
 if (strcmp (four,input_buf3) == 0)
{
   DCTIV_Row__InverseKernel_GPUAS <32> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);

   }

    plhs[0] = mxGPUCreateMxArrayOnGPU(B);   
    mxGPUDestroyGPUArray(B);
       
       }    
	}

    else if (!(mxIsGPUArray(prhs[0]))){
            
    if (mxGetClassID(prhs[0]) != mxSINGLE_CLASS) {
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Invalid input to MEX file, input(FIRST ARGUMENT) must be single precision (float).");
    } 
    if(mxIsComplex(prhs[0])){
       
       mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Incorrect input arguments!, input matrix must be real %s\n");
       
       }    
  	int numARows = (int)mxGetM(prhs[0]); 		// number of rows in the matrix A
	int numAColumns = (int)mxGetN(prhs[0]); 	// number of columns in the matrix A
	
	int numCRows;		// number of rows in the matrix C (you have to set this)
	int numCColumns;	// number of columns in the matrix C (you have to set this)
	
    if (numAColumns==1)
    {   
    printf("Attention, this is a column vector, please try Inverse Discrete Cosine Transform in column wise \n");
    return;
    }
    mxInitGPU();
    numCRows = numARows;
	numCColumns = numAColumns;
	float  * hostA ; // The A matrix
	hostA = (float *)mxGetData(prhs[0]);
    plhs[0] = mxCreateNumericMatrix(numCRows, numCColumns, mxSINGLE_CLASS, mxREAL);
    float  *pointer =(float*) mxGetPr(plhs[0]);
 
     if (strcmp (one,input_buf3) == 0)
{
      CalculateTransformDCTInverseRowOneS(hostA, pointer, numARows, numAColumns, numCRows, numCColumns);  
 } 
     if (strcmp (two,input_buf3) == 0)
{
      CalculateTransformDCTInverseRowTwoS(hostA, pointer, numARows, numAColumns, numCRows, numCColumns);  
 } 
     if (strcmp (three,input_buf3) == 0)
{
      CalculateTransformDCTInverseRowThreeS(hostA, pointer, numARows, numAColumns, numCRows, numCColumns);  
 } 
     if (strcmp (four,input_buf3) == 0)
{
      CalculateTransformDCTInverseRowFourS(hostA, pointer, numARows, numAColumns, numCRows, numCColumns);  
 } 

	
      }
    }
  } 
}


//SINE TRANSFORM
 if (strcmp (sine,input_buf1) == 0)
{

 if (strcmp (direct,input_buf2) == 0)
{   
  if (strcmp (column,input_buf4) == 0)
{   

    if (mxIsGPUArray(prhs[0])) {
	
    mxGPUArray const *A;
    mxGPUArray *B;
    float const *d_A;
    float *d_B;
    int numARows, numAColumns, numCRows,  numCColumns;
    mxInitGPU();
    cudaError_t error;
    int devID = 0;
    error = cudaGetDevice(&devID);
    cudaDeviceProp deviceProp;
     error = cudaGetDeviceProperties(&deviceProp, devID);
      if (error != cudaSuccess)
      {
          printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);
          exit(EXIT_FAILURE);
      }
    int TILEDIM = (deviceProp.major < 2) ? 16 : 32;

    A = mxGPUCreateFromMxArray(prhs[0]);
       if(mxGPUGetComplexity(A) != mxREAL){
       mxGPUDestroyGPUArray(A);
       mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Incorrect input arguments!, input matrix must be real %s\n");
       
       }
    const mwSize *dims;
   dims=mxGPUGetDimensions(A);
   numARows = (int)dims[0]; /* gets number of rows of A */
   numAColumns = (int)dims[1]; /* gets number of columns of A */
   size_t pivot_dimensA[2] = {numARows,numAColumns};
   mwSize NrOfDim=mxGPUGetNumberOfDimensions(A);
		numCRows = numARows;
		numCColumns = numAColumns;
        
  if (numARows==1)
 {   
 printf("Attention, this is a row vector, please try Discrete Sine Transform in row wise \n");
 return;
 }
 
    char const * const errId = "parallel:gpu:mexGPUExample:InvalidInput";
    char const * const errMsg = "Invalid input to MEX file, input(FIRST ARGUMENT) must be single precision (float).";

    if (mxGPUGetClassID(A) != mxSINGLE_CLASS) {
        mexErrMsgIdAndTxt(errId, errMsg);
    }

    d_A = (float const *)(mxGPUGetDataReadOnly(A));
    mxGPUDestroyGPUArray(A);
    B = mxGPUCreateGPUArray(NrOfDim, (mwSize*) pivot_dimensA, mxSINGLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    d_B = (float *)(mxGPUGetData(B));

    dim3 dimBlock;
	dim3 dimGrid;
    
     switch (TILEDIM){
        
        case 16:
            
     TILE_DIM= TILEDIM;
	 dimBlock.x=TILE_DIM;
	 dimBlock.y=TILE_DIM;
     dimBlock.z=1;
	dimGrid.x = (numCColumns + dimBlock.x - 1) / dimBlock.x;
	dimGrid.y = (numCRows + dimBlock.y - 1) / dimBlock.y;
 if (strcmp (one,input_buf3) == 0)
{
     
   DSTI_Column_Kernel_GPUAS <16> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
   
   
 }  
 if (strcmp (two,input_buf3) == 0)
{
     
   DSTII_Column_Kernel_GPUAS <16> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
   
   
 }
 if (strcmp (three,input_buf3) == 0)
{
     
   DSTIII_Column_Kernel_GPUAS <16> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
   
   
 }
 if (strcmp (four,input_buf3) == 0)
{
     
   DSTIV_Column_Kernel_GPUAS <16> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
   
   
 }

    plhs[0] = mxGPUCreateMxArrayOnGPU(B);  
    mxGPUDestroyGPUArray(B);
    
            case 32:
            
     TILE_DIM= TILEDIM;
	 dimBlock.x=TILE_DIM;
	 dimBlock.y=TILE_DIM;
     dimBlock.z=1;
	dimGrid.x = (numCColumns + dimBlock.x - 1) / dimBlock.x;
	dimGrid.y = (numCRows + dimBlock.y - 1) / dimBlock.y;
    
     if (strcmp (one,input_buf3) == 0)
{
     
   DSTI_Column_Kernel_GPUAS <32> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
   
   
 }  
 if (strcmp (two,input_buf3) == 0)
{
     
   DSTII_Column_Kernel_GPUAS <32> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
   
   
 }
 if (strcmp (three,input_buf3) == 0)
{
     
   DSTIII_Column_Kernel_GPUAS <32> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
   
   
 }
 if (strcmp (four,input_buf3) == 0)
{
     
   DSTIV_Column_Kernel_GPUAS <32> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
   
   
 }

    plhs[0] = mxGPUCreateMxArrayOnGPU(B);  
    mxGPUDestroyGPUArray(B);
     
       }   
	}

    else if (!(mxIsGPUArray(prhs[0]))){
            
    if (mxGetClassID(prhs[0]) != mxSINGLE_CLASS) {
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Invalid input to MEX file, input(FIRST ARGUMENT) must be single precision (float).");
    } 
    if(mxIsComplex(prhs[0])){
       
       mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Incorrect input arguments!, input matrix must be real %s\n");
       
       }
            
  	int numARows = (int)mxGetM(prhs[0]); 		// number of rows in the matrix A
	int numAColumns = (int)mxGetN(prhs[0]); 	// number of columns in the matrix A
	int numCRows;		// number of rows in the matrix C (you have to set this)
	int numCColumns;	// number of columns in the matrix C (you have to set this)
    	
		numCRows = numARows;
		numCColumns = numAColumns;
        
 if (numARows==1)
 {   
 printf("Attention, this is a row vector, please try Discrete Sine Transform in row wise \n");
 return;
 }
    mxInitGPU();
	float  * hostA ; // The A matrix
	hostA = (float *)mxGetData(prhs[0]);
    
    plhs[0] = mxCreateNumericMatrix(numCRows, numCColumns, mxSINGLE_CLASS, mxREAL);
    float  *pointer = (float*)mxGetPr(plhs[0]);
     
     if (strcmp (one,input_buf3) == 0)
    {
     
       CalculateTransformDSTColumnOneS(hostA, pointer, numARows, numAColumns, numCRows, numCColumns);
	
    }
     if (strcmp (two,input_buf3) == 0)
    {
     
       CalculateTransformDSTColumnTwoS(hostA, pointer, numARows, numAColumns, numCRows, numCColumns);
	
    }
     if (strcmp (three,input_buf3) == 0)
    {
     
       CalculateTransformDSTColumnThreeS(hostA, pointer, numARows, numAColumns, numCRows, numCColumns);
	
    }
     if (strcmp (four,input_buf3) == 0)
    {
     
       CalculateTransformDSTColumnFourS(hostA, pointer, numARows, numAColumns, numCRows, numCColumns);
	
    }

      } 
    } 
  
     if (strcmp (row,input_buf4) == 0)
    {
    
    if (mxIsGPUArray(prhs[0])) {
		
    mxGPUArray const *A;
    mxGPUArray *B;
    float const *d_A;
    float *d_B;
    int numARows, numAColumns, numCRows,  numCColumns;
    mxInitGPU();
    cudaError_t error;
    int devID = 0;
    error = cudaGetDevice(&devID);
    cudaDeviceProp deviceProp;
     error = cudaGetDeviceProperties(&deviceProp, devID);
      if (error != cudaSuccess)
      {
          printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);
          exit(EXIT_FAILURE);
      }
    int TILEDIM = (deviceProp.major < 2) ? 16 : 32;

    A = mxGPUCreateFromMxArray(prhs[0]);
       if(mxGPUGetComplexity(A) != mxREAL){
       mxGPUDestroyGPUArray(A);
       mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Incorrect input arguments!, input matrix must be real %s\n");
       
       }
   const mwSize *dims;
   dims=mxGPUGetDimensions(A);
   numARows = (int)dims[0]; /* gets number of rows of A */
   numAColumns = (int)dims[1]; /* gets number of columns of A */
   size_t pivot_dimensA[2] = {numARows,numAColumns};
   mwSize NrOfDim=mxGPUGetNumberOfDimensions(A);
   
 if (numAColumns==1)
 {   
 printf("Attention, this is a column vector, please try Discrete Sine Transform in column wise \n");
 return;
 }
    numCRows = numARows;
	numCColumns = numAColumns;

    char const * const errId = "parallel:gpu:mexGPUExample:InvalidInput";
    char const * const errMsg = "Invalid input to MEX file, input(FIRST ARGUMENT) must be single precision (float).";
    
    if (mxGPUGetClassID(A) != mxSINGLE_CLASS) {
        mexErrMsgIdAndTxt(errId, errMsg);
    }

    d_A = (float const *)(mxGPUGetDataReadOnly(A));
    mxGPUDestroyGPUArray(A);
    B = mxGPUCreateGPUArray(NrOfDim, (mwSize*) pivot_dimensA, mxSINGLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    d_B = (float *)(mxGPUGetData(B));

    dim3 dimBlock;
	dim3 dimGrid;
    
     switch (TILEDIM){
        
        case 16:
            
     TILE_DIM= TILEDIM;
	 dimBlock.x=TILE_DIM;
	 dimBlock.y=TILE_DIM;
     dimBlock.z=1;
	 dimGrid.x = (numCColumns + dimBlock.x - 1) / dimBlock.x;
	 dimGrid.y = (numCRows + dimBlock.y - 1) / dimBlock.y;
 if (strcmp (one,input_buf3) == 0)
{
    
   DSTI_Row_Kernel_GPUAS <16> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
}
 if (strcmp (two,input_buf3) == 0)
{
    
   DSTII_Row_Kernel_GPUAS <16> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
}
 if (strcmp (three,input_buf3) == 0)
{
    
   DSTIII_Row_Kernel_GPUAS <16> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
}
 if (strcmp (four,input_buf3) == 0)
{
    
   DSTIV_Row_Kernel_GPUAS <16> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
}
   
    plhs[0] = mxGPUCreateMxArrayOnGPU(B);  
    mxGPUDestroyGPUArray(B);
     
      case 32:
            
     TILE_DIM= TILEDIM;
	 dimBlock.x=TILE_DIM;
	 dimBlock.y=TILE_DIM;
     dimBlock.z=1;
	 dimGrid.x = (numCColumns + dimBlock.x - 1) / dimBlock.x;
	 dimGrid.y = (numCRows + dimBlock.y - 1) / dimBlock.y;
     if (strcmp (one,input_buf3) == 0)
{
    
   DSTI_Row_Kernel_GPUAS <32> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
}
 if (strcmp (two,input_buf3) == 0)
{
    
   DSTII_Row_Kernel_GPUAS <32> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
}
 if (strcmp (three,input_buf3) == 0)
{
    
   DSTIII_Row_Kernel_GPUAS <32> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
}
 if (strcmp (four,input_buf3) == 0)
{
    
   DSTIV_Row_Kernel_GPUAS <32> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
}
     
    plhs[0] = mxGPUCreateMxArrayOnGPU(B);   
    mxGPUDestroyGPUArray(B);
    
      }
	}
 
    else if (!(mxIsGPUArray(prhs[0]))){
            
    if (mxGetClassID(prhs[0]) != mxSINGLE_CLASS) {
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Invalid input to MEX file, input(FIRST ARGUMENT) must be single precision (float).");
    } 
    if(mxIsComplex(prhs[0])){
       
       mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Incorrect input arguments!, input matrix must be real %s\n");
       
       }
       
  	int numARows = (int)mxGetM(prhs[0]); 		// number of rows in the matrix A
	int numAColumns = (int)mxGetN(prhs[0]); 	// number of columns in the matrix A
	
	int numCRows;		// number of rows in the matrix C (you have to set this)
	int numCColumns;	// number of columns in the matrix C (you have to set this)
	
	numCRows = numARows;
	numCColumns = numAColumns;
	float  * hostA ; // The A matrix
	
 if (numAColumns==1)
 {   
 printf("Attention, this is a column vector, please try Discrete Sine Transform in column wise \n");
 return;
 }
   
    mxInitGPU();
	hostA = (float *)mxGetData(prhs[0]);
    plhs[0] = mxCreateNumericMatrix(numCRows, numCColumns, mxSINGLE_CLASS, mxREAL);
    float  *pointer = (float*)mxGetPr(plhs[0]);
    
     if (strcmp (one,input_buf3) == 0)
{  
      CalculateTransformDSTRowOneS(hostA, pointer, numARows, numAColumns, numCRows, numCColumns);
      
      } 
       if (strcmp (two,input_buf3) == 0)
{  
      CalculateTransformDSTRowTwoS(hostA, pointer, numARows, numAColumns, numCRows, numCColumns);
      
      }
       if (strcmp (three,input_buf3) == 0)
{  
      CalculateTransformDSTRowThreeS(hostA, pointer, numARows, numAColumns, numCRows, numCColumns);
      
      }
       if (strcmp (four,input_buf3) == 0)
{  
      CalculateTransformDSTRowFourS(hostA, pointer, numARows, numAColumns, numCRows, numCColumns);
      
      }
  
      } 
   }  
}

    if (strcmp (inverse,input_buf2) == 0)
     {
    if (strcmp (column,input_buf4) == 0)
     {      
    if (mxIsGPUArray(prhs[0])) {

    mxGPUArray const *A;
    mxGPUArray *B;
    float const *d_A;
    float *d_B;
    int numARows, numAColumns, numCRows,  numCColumns;
    mxInitGPU();
    cudaError_t error;
    int devID = 0;
    error = cudaGetDevice(&devID);
    cudaDeviceProp deviceProp;
     error = cudaGetDeviceProperties(&deviceProp, devID);
      if (error != cudaSuccess)
      {
          printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);
          exit(EXIT_FAILURE);
      }
    int TILEDIM = (deviceProp.major < 2) ? 16 : 32;

    A = mxGPUCreateFromMxArray(prhs[0]);
       if(mxGPUGetComplexity(A) != mxREAL){
       mxGPUDestroyGPUArray(A);
       mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Incorrect input arguments!, input matrix must be real %s\n");
       
       }
    const mwSize *dims;
    dims=mxGPUGetDimensions(A);
    numARows = (int)dims[0]; /* gets number of rows of A */
    numAColumns = (int)dims[1]; /* gets number of columns of A */
    size_t pivot_dimensA[2] = {numARows,numAColumns};
    mwSize NrOfDim=mxGPUGetNumberOfDimensions(A);
    numCRows = numARows;
    numCColumns = numAColumns;
   
  if (numARows==1)
 {   
 printf("Attention, this is a row vector, please try Inverse Discrete Sine Transform in row wise \n");
 return;
 }
 
    char const * const errId = "parallel:gpu:mexGPUExample:InvalidInput";
    char const * const errMsg = "Invalid input to MEX file, input(FIRST ARGUMENT) must be single precision (float).";

    if (mxGPUGetClassID(A) != mxSINGLE_CLASS) {
        mexErrMsgIdAndTxt(errId, errMsg);
    }
    d_A = (float const *)(mxGPUGetDataReadOnly(A));
    mxGPUDestroyGPUArray(A);
    B = mxGPUCreateGPUArray(NrOfDim, (mwSize*) pivot_dimensA, mxSINGLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);    
    d_B = (float *)(mxGPUGetData(B));
    
    dim3 dimBlock;
	dim3 dimGrid;
    
     switch (TILEDIM){
        
        case 16:
            
     TILE_DIM= TILEDIM;
	 dimBlock.x=TILE_DIM;
	 dimBlock.y=TILE_DIM;
     dimBlock.z=1;
	 dimGrid.x = (numCColumns + dimBlock.x - 1) / dimBlock.x;
	 dimGrid.y = (numCRows + dimBlock.y - 1) / dimBlock.y;
 if (strcmp (one,input_buf3) == 0)
{
    
   DSTI_Column_Inverse_Kernel_GPUAS <16> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
   
   
}
  if (strcmp (two,input_buf3) == 0)
{
    
   DSTII_Column_Inverse_Kernel_GPUAS <16> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
   
   
}
 if (strcmp (three,input_buf3) == 0)
{
    
   DSTIII_Column_Inverse_Kernel_GPUAS <16> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
   
   
}
 if (strcmp (four,input_buf3) == 0)
{
    
   DSTIV_Column_Inverse_Kernel_GPUAS <16> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
   
   
}

    plhs[0] = mxGPUCreateMxArrayOnGPU(B);     
    mxGPUDestroyGPUArray(B);
    
      case 32:
            
     TILE_DIM= TILEDIM;
	 dimBlock.x=TILE_DIM;
	 dimBlock.y=TILE_DIM;
     dimBlock.z=1;
	 dimGrid.x = (numCColumns + dimBlock.x - 1) / dimBlock.x;
	 dimGrid.y = (numCRows + dimBlock.y - 1) / dimBlock.y; 
    
     if (strcmp (one,input_buf3) == 0)
{
    
   DSTI_Column_Inverse_Kernel_GPUAS <32> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
   
   
}
  if (strcmp (two,input_buf3) == 0)
{
    
   DSTII_Column_Inverse_Kernel_GPUAS <32> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
   
   
}
 if (strcmp (three,input_buf3) == 0)
{
    
   DSTIII_Column_Inverse_Kernel_GPUAS <32> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
   
   
}
 if (strcmp (four,input_buf3) == 0)
{
    
   DSTIV_Column_Inverse_Kernel_GPUAS <32> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
   
   
}

    plhs[0] = mxGPUCreateMxArrayOnGPU(B);         
    mxGPUDestroyGPUArray(B);
    
      }     
	}

    else if (!(mxIsGPUArray(prhs[0]))){
            
    if (mxGetClassID(prhs[0]) != mxSINGLE_CLASS) {
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Invalid input to MEX file, input(FIRST ARGUMENT) must be single precision (float).");
    }    
    if(mxIsComplex(prhs[0])){
       
       mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Incorrect input arguments!, input matrix must be real %s\n");
       
       }
       
  	int numARows = (int)mxGetM(prhs[0]); 		// number of rows in the matrix A
	int numAColumns = (int)mxGetN(prhs[0]); 	// number of columns in the matrix A
	int numCRows;		// number of rows in the matrix C (you have to set this)
	int numCColumns;	// number of columns in the matrix C (you have to set this)
    	
		numCRows = numARows;
		numCColumns = numAColumns;
        
 if (numARows==1)
 {   
 printf("Attention, this is a row vector, please try Inverse Discrete Sine Transform in row wise \n");
 return;
 }
    mxInitGPU();
	float  * hostA ; // The A matrix
	hostA = (float *)mxGetData(prhs[0]);
    
    plhs[0] = mxCreateNumericMatrix(numCRows, numCColumns, mxSINGLE_CLASS, mxREAL);
    float  *pointer = (float*)mxGetPr(plhs[0]);
    
     if (strcmp (one,input_buf3) == 0)
   {
  
       CalculateTransformDSTInverseColumnOneS(hostA, pointer, numARows, numAColumns, numCRows, numCColumns);
	
   }
     if (strcmp (two,input_buf3) == 0)
   {
  
       CalculateTransformDSTInverseColumnTwoS(hostA, pointer, numARows, numAColumns, numCRows, numCColumns);
	
   }
     if (strcmp (three,input_buf3) == 0)
   {
  
       CalculateTransformDSTInverseColumnThreeS(hostA, pointer, numARows, numAColumns, numCRows, numCColumns);
	
   }
     if (strcmp (four,input_buf3) == 0)
   {
  
       CalculateTransformDSTInverseColumnFourS(hostA, pointer, numARows, numAColumns, numCRows, numCColumns);
	
   }

   } 
} 

    if (strcmp (row,input_buf4) == 0)
    {
    
    if (mxIsGPUArray(prhs[0])) {

    mxGPUArray const *A;
    mxGPUArray *B;
    float const *d_A;
    float *d_B;
    int numARows, numAColumns, numCRows,  numCColumns;
    mxInitGPU();
    cudaError_t error;
    int devID = 0;
    error = cudaGetDevice(&devID);
    cudaDeviceProp deviceProp;
     error = cudaGetDeviceProperties(&deviceProp, devID);
    if (error != cudaSuccess)
      {
          printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);
          exit(EXIT_FAILURE);
      }
    int TILEDIM = (deviceProp.major < 2) ? 16 : 32;  

    A = mxGPUCreateFromMxArray(prhs[0]);
       if(mxGPUGetComplexity(A) != mxREAL){
       mxGPUDestroyGPUArray(A);
       mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Incorrect input arguments!, input matrix must be real %s\n");
       
       }
    const mwSize *dims;
    dims=mxGPUGetDimensions(A);
    numARows = (int)dims[0]; /* gets number of rows of A */
    numAColumns = (int)dims[1]; /* gets number of columns of A */
    size_t pivot_dimensA[2] = {numARows,numAColumns};
    mwSize NrOfDim=mxGPUGetNumberOfDimensions(A);
    
    if (numAColumns==1)
 {   
 printf("Attention, this is a column vector, please try Inverse Discrete Sine Transform in column wise \n");
 return;
 }
 
    numCRows = numARows;
	numCColumns = numAColumns;
    
    char const * const errId = "parallel:gpu:mexGPUExample:InvalidInput";
    char const * const errMsg = "Invalid input to MEX file, input(FIRST ARGUMENT) must be single precision (float).";

    if (mxGPUGetClassID(A) != mxSINGLE_CLASS) {
        mexErrMsgIdAndTxt(errId, errMsg);
    }
    
    d_A = (float const *)(mxGPUGetDataReadOnly(A));    
    mxGPUDestroyGPUArray(A);
    B = mxGPUCreateGPUArray(NrOfDim, (mwSize*) pivot_dimensA, mxSINGLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE); 
    d_B = (float *)(mxGPUGetData(B));
    
    dim3 dimBlock;
	dim3 dimGrid;
    
     switch (TILEDIM){
        
        case 16:
            
     TILE_DIM= TILEDIM;
	 dimBlock.x=TILE_DIM;
	 dimBlock.y=TILE_DIM;
     dimBlock.z=1;
	 dimGrid.x = (numCColumns + dimBlock.x - 1) / dimBlock.x;
	 dimGrid.y = (numCRows + dimBlock.y - 1) / dimBlock.y;
 if (strcmp (one,input_buf3) == 0)
{
   DSTI_Row__InverseKernel_GPUAS <16> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);

   }
 if (strcmp (two,input_buf3) == 0)
{
   DSTII_Row__InverseKernel_GPUAS <16> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);

   }
 if (strcmp (three,input_buf3) == 0)
{
   DSTIII_Row__InverseKernel_GPUAS <16> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);

   }
 if (strcmp (four,input_buf3) == 0)
{
   DSTIV_Row__InverseKernel_GPUAS <16> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);

   }

    plhs[0] = mxGPUCreateMxArrayOnGPU(B);        
    mxGPUDestroyGPUArray(B);
    
          case 32:
            
     TILE_DIM= TILEDIM;
	 dimBlock.x=TILE_DIM;
	 dimBlock.y=TILE_DIM;
     dimBlock.z=1;
	 dimGrid.x = (numCColumns + dimBlock.x - 1) / dimBlock.x;
	 dimGrid.y = (numCRows + dimBlock.y - 1) / dimBlock.y;
    
    if (strcmp (one,input_buf3) == 0)
{
   DSTI_Row__InverseKernel_GPUAS <32> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);

   }
 if (strcmp (two,input_buf3) == 0)
{
   DSTII_Row__InverseKernel_GPUAS <32> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);

   }
 if (strcmp (three,input_buf3) == 0)
{
   DSTIII_Row__InverseKernel_GPUAS <32> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);

   }
 if (strcmp (four,input_buf3) == 0)
{
   DSTIV_Row__InverseKernel_GPUAS <32> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);

   }

    plhs[0] = mxGPUCreateMxArrayOnGPU(B);    
    mxGPUDestroyGPUArray(B); 
        
      }     
	}
 
    else if (!(mxIsGPUArray(prhs[0]))){
            
    if (mxGetClassID(prhs[0]) != mxSINGLE_CLASS) {
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Invalid input to MEX file, input(FIRST ARGUMENT) must be single precision (float).");
    }  
    if(mxIsComplex(prhs[0])){
       
       mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Incorrect input arguments!, input matrix must be real %s\n");
       
       }    
    
  	int numARows = (int)mxGetM(prhs[0]); 		// number of rows in the matrix A
	int numAColumns = (int)mxGetN(prhs[0]); 	// number of columns in the matrix A
	
	int numCRows;		// number of rows in the matrix C (you have to set this)
	int numCColumns;	// number of columns in the matrix C (you have to set this)
		
    if (numAColumns==1)
    {   
    printf("Attention, this is a column vector, please try Inverse Discrete Sine Transform in column wise \n");
    return;
    }
    mxInitGPU();
    numCRows = numARows;
	numCColumns = numAColumns;
	float  * hostA ; // The A matrix
	hostA = (float *)mxGetData(prhs[0]);
    
    plhs[0] = mxCreateNumericMatrix(numCRows, numCColumns, mxSINGLE_CLASS, mxREAL);
    float  *pointer =(float*) mxGetPr(plhs[0]);
    
     if (strcmp (one,input_buf3) == 0)
{
      CalculateTransformDSTInverseRowOneS(hostA, pointer, numARows, numAColumns, numCRows, numCColumns);  
 } 
     if (strcmp (two,input_buf3) == 0)
{
      CalculateTransformDSTInverseRowTwoS(hostA, pointer, numARows, numAColumns, numCRows, numCColumns);  
 } 
     if (strcmp (three,input_buf3) == 0)
{
      CalculateTransformDSTInverseRowThreeS(hostA, pointer, numARows, numAColumns, numCRows, numCColumns);  
 } 
     if (strcmp (four,input_buf3) == 0)
{
      CalculateTransformDSTInverseRowFourS(hostA, pointer, numARows, numAColumns, numCRows, numCColumns);  
 } 

      }
    }
  } 
}

}
