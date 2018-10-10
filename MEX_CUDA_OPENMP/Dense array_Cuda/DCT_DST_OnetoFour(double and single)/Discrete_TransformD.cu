
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
#include "CuFilesD/Discrete_Transform_kernel.cuh"
#include "CuFilesD/DCT_I_Column.cu"
#include "CuFilesD/DCT_I_Row.cu"
#include "CuFilesD/DCT_I_Column_Inverse.cu"
#include "CuFilesD/DCT_I_Row_Inverse.cu"
#include "CuFilesD/DCT_II_Row.cu"
#include "CuFilesD/DCT_II_Row_Inverse.cu"
#include "CuFilesD/DCT_II_Column.cu"
#include "CuFilesD/DCT_II_Column_Inverse.cu"
#include "CuFilesD/DCT_III_Row.cu"
#include "CuFilesD/DCT_III_Row_Inverse.cu"
#include "CuFilesD/DCT_III_Column.cu"
#include "CuFilesD/DCT_III_Column_Inverse.cu"
#include "CuFilesD/DCT_IV_Row.cu"
#include "CuFilesD/DCT_IV_Row_Inverse.cu"
#include "CuFilesD/DCT_IV_Column.cu"
#include "CuFilesD/DCT_IV_Column_Inverse.cu"
#include "CuFilesD/DST_I_Column.cu"
#include "CuFilesD/DST_I_Row.cu"
#include "CuFilesD/DST_I_Column_Inverse.cu"
#include "CuFilesD/DST_I_Row_Inverse.cu"
#include "CuFilesD/DST_II_Row.cu"
#include "CuFilesD/DST_II_Row_Inverse.cu"
#include "CuFilesD/DST_II_Column.cu"
#include "CuFilesD/DST_II_Column_Inverse.cu"
#include "CuFilesD/DST_III_Row.cu"
#include "CuFilesD/DST_III_Row_Inverse.cu"
#include "CuFilesD/DST_III_Column.cu"
#include "CuFilesD/DST_III_Column_Inverse.cu"
#include "CuFilesD/DST_IV_Row.cu"
#include "CuFilesD/DST_IV_Row_Inverse.cu"
#include "CuFilesD/DST_IV_Column.cu"
#include "CuFilesD/DST_IV_Column_Inverse.cu"
//#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#define DEFAULT_DIM 32   
#define 	DELTA(i, j)   ((i==j)?1:0)
//#define TILE_DIM 16
unsigned int TILE_DIM=16;
// DCT
extern "C" void  CalculateTransformDCTColumnOne(double * A, double * C, int numARows,
	int numAColumns, int numCRows, int numCColumns);
    
extern "C" void  CalculateTransformDCTInverseColumnOne(double * A, double * C, int numARows,
	int numAColumns, int numCRows, int numCColumns);
    
extern "C" void  CalculateTransformDCTRowOne(double * A, double * C, int numARows,
	int numAColumns, int numCRows, int numCColumns);
    
extern "C" void  CalculateTransformDCTInverseRowOne(double * A, double * C, int numARows,
	int numAColumns, int numCRows, int numCColumns);    
    
extern "C" void  CalculateTransformDCTRowTwo(double * A, double * C, int numARows,
	int numAColumns, int numCRows, int numCColumns);
    
extern "C" void  CalculateTransformDCTInverseRowTwo(double * A, double * C, int numARows,
	int numAColumns, int numCRows, int numCColumns);
    
extern "C" void  CalculateTransformDCTColumnTwo(double * A, double * C, int numARows,
	int numAColumns, int numCRows, int numCColumns);
    
extern "C" void  CalculateTransformDCTInverseColumnTwo(double * A, double * C, int numARows,
	int numAColumns, int numCRows, int numCColumns);
    
extern "C" void  CalculateTransformDCTColumnThree(double * A, double * C, int numARows,
	int numAColumns, int numCRows, int numCColumns); 
    
extern "C" void  CalculateTransformDCTInverseColumnThree(double * A, double * C, int numARows,
	int numAColumns, int numCRows, int numCColumns);
    
extern "C" void  CalculateTransformDCTRowThree(double * A, double * C, int numARows,
	int numAColumns, int numCRows, int numCColumns);
    
extern "C" void  CalculateTransformDCTInverseRowThree(double * A, double * C, int numARows,
	int numAColumns, int numCRows, int numCColumns);
    
extern "C" void  CalculateTransformDCTColumnFour(double * A, double * C, int numARows,
	int numAColumns, int numCRows, int numCColumns);  
    
extern "C" void  CalculateTransformDCTInverseColumnFour(double * A, double * C, int numARows,
	int numAColumns, int numCRows, int numCColumns);
    
extern "C" void  CalculateTransformDCTRowFour(double * A, double * C, int numARows,
	int numAColumns, int numCRows, int numCColumns);
    
extern "C" void  CalculateTransformDCTInverseRowFour(double * A, double * C, int numARows,
	int numAColumns, int numCRows, int numCColumns); 
    
    // DST
extern "C" void  CalculateTransformDSTColumnOne(double * A, double * C, int numARows,
	int numAColumns, int numCRows, int numCColumns);
    
extern "C" void  CalculateTransformDSTInverseColumnOne(double * A, double * C, int numARows,
	int numAColumns, int numCRows, int numCColumns);
    
extern "C" void  CalculateTransformDSTRowOne(double * A, double * C, int numARows,
	int numAColumns, int numCRows, int numCColumns);
    
extern "C" void  CalculateTransformDSTInverseRowOne(double * A, double * C, int numARows,
	int numAColumns, int numCRows, int numCColumns);    
    
extern "C" void  CalculateTransformDSTRowTwo(double * A, double * C, int numARows,
	int numAColumns, int numCRows, int numCColumns);
    
extern "C" void  CalculateTransformDSTInverseRowTwo(double * A, double * C, int numARows,
	int numAColumns, int numCRows, int numCColumns);
    
extern "C" void  CalculateTransformDSTColumnTwo(double * A, double * C, int numARows,
	int numAColumns, int numCRows, int numCColumns);
    
extern "C" void  CalculateTransformDSTInverseColumnTwo(double * A, double * C, int numARows,
	int numAColumns, int numCRows, int numCColumns);
    
extern "C" void  CalculateTransformDSTColumnThree(double * A, double * C, int numARows,
	int numAColumns, int numCRows, int numCColumns); 
    
extern "C" void  CalculateTransformDSTInverseColumnThree(double * A, double * C, int numARows,
	int numAColumns, int numCRows, int numCColumns);
    
extern "C" void  CalculateTransformDSTRowThree(double * A, double * C, int numARows,
	int numAColumns, int numCRows, int numCColumns);
    
extern "C" void  CalculateTransformDSTInverseRowThree(double * A, double * C, int numARows,
	int numAColumns, int numCRows, int numCColumns);
    
extern "C" void  CalculateTransformDSTColumnFour(double * A, double * C, int numARows,
	int numAColumns, int numCRows, int numCColumns);  
    
extern "C" void  CalculateTransformDSTInverseColumnFour(double * A, double * C, int numARows,
	int numAColumns, int numCRows, int numCColumns);
    
extern "C" void  CalculateTransformDSTRowFour(double * A, double * C, int numARows,
	int numAColumns, int numCRows, int numCColumns);
    
extern "C" void  CalculateTransformDSTInverseRowFour(double * A, double * C, int numARows,
	int numAColumns, int numCRows, int numCColumns); 
    
extern "C" static void mexTransD(int nlhs, mxArray *plhs[],
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
 
    char const * const InputErrMsg = "Invalid input to MEX file, input(FIRST ARGUMENT) must be double  precision (double), and the number of input arguments must be five.";
    
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
    double const *d_A;
    double *d_B;
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
    char const * const errMsg = "Invalid input to MEX file, input(FIRST ARGUMENT) must be double  precision (double).";

    if (mxGPUGetClassID(A) != mxDOUBLE_CLASS) {
        mexErrMsgIdAndTxt(errId, errMsg);
    }

    d_A = (double const *)(mxGPUGetDataReadOnly(A));
    mxGPUDestroyGPUArray(A);
    B = mxGPUCreateGPUArray(NrOfDim, (mwSize*) pivot_dimensA, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    d_B = (double *)(mxGPUGetData(B));
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
     
   DCTI_Column_Kernel_GPUA <16> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
   
   
 }  
 if (strcmp (two,input_buf3) == 0)
{
     
   DCTII_Column_Kernel_GPUA <16> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
   
   
 }
 if (strcmp (three,input_buf3) == 0)
{
     
   DCTIII_Column_Kernel_GPUA <16> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
   
   
 }
 if (strcmp (four,input_buf3) == 0)
{
     
   DCTIV_Column_Kernel_GPUA <16> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
   
   
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
     
   DCTI_Column_Kernel_GPUA <32> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
   
   
 }  
 if (strcmp (two,input_buf3) == 0)
{
     
   DCTII_Column_Kernel_GPUA <32> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
   
   
 }
 if (strcmp (three,input_buf3) == 0)
{
     
   DCTIII_Column_Kernel_GPUA <32> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
   
   
 }
 if (strcmp (four,input_buf3) == 0)
{
     
   DCTIV_Column_Kernel_GPUA <32> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
   
   
 }

    plhs[0] = mxGPUCreateMxArrayOnGPU(B);  
    mxGPUDestroyGPUArray(B);
    
      } 
    }

    else if (!(mxIsGPUArray(prhs[0]))){
             
   if (mxGetClassID(prhs[0]) != mxDOUBLE_CLASS) {
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Invalid input to MEX file, input(FIRST ARGUMENT) must be double  precision (double).");
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
	double  * hostA ; // The A matrix
	hostA = (double *)mxGetData(prhs[0]);   
    plhs[0] = mxCreateNumericMatrix(numCRows, numCColumns, mxDOUBLE_CLASS, mxREAL);
    double  *pointer =(double*) mxGetPr(plhs[0]);
  
     if (strcmp (one,input_buf3) == 0)
    {
     
       CalculateTransformDCTColumnOne(hostA, pointer, numARows, numAColumns, numCRows, numCColumns);
	
    }
     if (strcmp (two,input_buf3) == 0)
    {
     
       CalculateTransformDCTColumnTwo(hostA, pointer, numARows, numAColumns, numCRows, numCColumns);
       
	
    }
     if (strcmp (three,input_buf3) == 0)
    {
     
       CalculateTransformDCTColumnThree(hostA, pointer, numARows, numAColumns, numCRows, numCColumns);
	
    }
     if (strcmp (four,input_buf3) == 0)
    {
     
       CalculateTransformDCTColumnFour(hostA, pointer, numARows, numAColumns, numCRows, numCColumns);
	
         }	
       } 
    } 
   
     if (strcmp (row,input_buf4) == 0)
{

    if (mxIsGPUArray(prhs[0])) {
		
    mxGPUArray const *A;
    mxGPUArray *B;
    double const *d_A;
    double *d_B;
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
   char const * const errMsg = "Invalid input to MEX file, input(FIRST ARGUMENT) must be double  precision (double).";
    if (mxGPUGetClassID(A) != mxDOUBLE_CLASS) {
        mexErrMsgIdAndTxt(errId, errMsg);
    }
    
    d_A = (double const *)(mxGPUGetDataReadOnly(A));
    mxGPUDestroyGPUArray(A);
    B = mxGPUCreateGPUArray(NrOfDim, (mwSize*) pivot_dimensA, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    d_B = (double *)(mxGPUGetData(B));
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
    
   DCTI_Row_Kernel_GPUA <16> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
}
 if (strcmp (two,input_buf3) == 0)
{
    
   DCTII_Row_Kernel_GPUA <16> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
}
 if (strcmp (three,input_buf3) == 0)
{
    
   DCTIII_Row_Kernel_GPUA <16> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
}
 if (strcmp (four,input_buf3) == 0)
{
    
   DCTIV_Row_Kernel_GPUA <16> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
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
    
   DCTI_Row_Kernel_GPUA <32> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
}
 if (strcmp (two,input_buf3) == 0)
{
    
   DCTII_Row_Kernel_GPUA <32> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
}
 if (strcmp (three,input_buf3) == 0)
{
    
   DCTIII_Row_Kernel_GPUA <32> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
}
 if (strcmp (four,input_buf3) == 0)
{
    
   DCTIV_Row_Kernel_GPUA <32> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
}

    plhs[0] = mxGPUCreateMxArrayOnGPU(B);
    mxGPUDestroyGPUArray(B);
     
        }    
	}

    else if (!(mxIsGPUArray(prhs[0]))){
            
   if (mxGetClassID(prhs[0]) != mxDOUBLE_CLASS) {
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Invalid input to MEX file, input(FIRST ARGUMENT) must be double  precision (double).");
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
	double  * hostA ; // The A matrix
	
 if (numAColumns==1)
 {   
 printf("Attention, this is a column vector, please try Discrete Cosine Transform in column wise \n");
 return;
 }

    mxInitGPU();
	hostA = (double *)mxGetData(prhs[0]);
    plhs[0] = mxCreateNumericMatrix(numCRows, numCColumns, mxDOUBLE_CLASS, mxREAL);
    double  *pointer = (double*)mxGetPr(plhs[0]);
     if (strcmp (one,input_buf3) == 0)
{  
      CalculateTransformDCTRowOne(hostA, pointer, numARows, numAColumns, numCRows, numCColumns);
      
      } 
       if (strcmp (two,input_buf3) == 0)
{  
      CalculateTransformDCTRowTwo(hostA, pointer, numARows, numAColumns, numCRows, numCColumns);
      
      }
       if (strcmp (three,input_buf3) == 0)
{  
      CalculateTransformDCTRowThree(hostA, pointer, numARows, numAColumns, numCRows, numCColumns);
      
      }
       if (strcmp (four,input_buf3) == 0)
{  
      CalculateTransformDCTRowFour(hostA, pointer, numARows, numAColumns, numCRows, numCColumns);
      
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
    double const *d_A;
    double *d_B;
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
    char const * const errMsg = "Invalid input to MEX file, input(FIRST ARGUMENT) must be double  precision (double).";

    if (mxGPUGetClassID(A) != mxDOUBLE_CLASS) {
        mexErrMsgIdAndTxt(errId, errMsg);
    }

    d_A = (double const *)(mxGPUGetDataReadOnly(A));
    mxGPUDestroyGPUArray(A);
    B = mxGPUCreateGPUArray(NrOfDim, (mwSize*) pivot_dimensA, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    d_B = (double *)(mxGPUGetData(B));
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
    
   DCTI_Column_Inverse_Kernel_GPUA <16> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
   
   
}
  if (strcmp (two,input_buf3) == 0)
{
    
   DCTII_Column_Inverse_Kernel_GPUA <16> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
   
   
}
 if (strcmp (three,input_buf3) == 0)
{
    
   DCTIII_Column_Inverse_Kernel_GPUA <16> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
   
   
}
 if (strcmp (four,input_buf3) == 0)
{
    
   DCTIV_Column_Inverse_Kernel_GPUA <16> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
   
   
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
    
   DCTI_Column_Inverse_Kernel_GPUA <32> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
   
   
}
  if (strcmp (two,input_buf3) == 0)
{
    
   DCTII_Column_Inverse_Kernel_GPUA <32> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
   
   
}
 if (strcmp (three,input_buf3) == 0)
{
    
   DCTIII_Column_Inverse_Kernel_GPUA <32> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
   
   
}
 if (strcmp (four,input_buf3) == 0)
{
    
   DCTIV_Column_Inverse_Kernel_GPUA <32> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
   
   
}
 
    plhs[0] = mxGPUCreateMxArrayOnGPU(B);   
    mxGPUDestroyGPUArray(B);
    
       }   
	}

    else if (!(mxIsGPUArray(prhs[0]))){
  
    if (mxGetClassID(prhs[0]) != mxDOUBLE_CLASS) {
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Invalid input to MEX file, input(FIRST ARGUMENT) must be double  precision (double).");
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

	double  * hostA ; // The A matrix
	hostA = (double *)mxGetData(prhs[0]);
    plhs[0] = mxCreateNumericMatrix(numCRows, numCColumns, mxDOUBLE_CLASS, mxREAL);
    double  *pointer = (double*)mxGetPr(plhs[0]);
     
     if (strcmp (one,input_buf3) == 0)
   {
  
       CalculateTransformDCTInverseColumnOne(hostA, pointer, numARows, numAColumns, numCRows, numCColumns);
	
   }
     if (strcmp (two,input_buf3) == 0)
   {
  
       CalculateTransformDCTInverseColumnTwo(hostA, pointer, numARows, numAColumns, numCRows, numCColumns);
	
   }
     if (strcmp (three,input_buf3) == 0)
   {
  
       CalculateTransformDCTInverseColumnThree(hostA, pointer, numARows, numAColumns, numCRows, numCColumns);
	
   }
     if (strcmp (four,input_buf3) == 0)
   {
  
       CalculateTransformDCTInverseColumnFour(hostA, pointer, numARows, numAColumns, numCRows, numCColumns);
	
      }	
   } 
}

    if (strcmp (row,input_buf4) == 0)
{
     
    if (mxIsGPUArray(prhs[0])) {

    mxGPUArray const *A;
    mxGPUArray *B;
    double const *d_A;
   
    double *d_B;
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
    char const * const errMsg = "Invalid input to MEX file, input(FIRST ARGUMENT) must be double  precision (double).";

    if (mxGPUGetClassID(A) != mxDOUBLE_CLASS) {
        mexErrMsgIdAndTxt(errId, errMsg);
    }

    d_A = (double const *)(mxGPUGetDataReadOnly(A));
    mxGPUDestroyGPUArray(A);
    B = mxGPUCreateGPUArray(NrOfDim, (mwSize*) pivot_dimensA, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    d_B = (double *)(mxGPUGetData(B));
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
   DCTI_Row__InverseKernel_GPUA <16> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);

   }
 if (strcmp (two,input_buf3) == 0)
{
   DCTII_Row__InverseKernel_GPUA <16> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);

   }
 if (strcmp (three,input_buf3) == 0)
{
   DCTIII_Row__InverseKernel_GPUA <16> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);

   }
 if (strcmp (four,input_buf3) == 0)
{
   DCTIV_Row__InverseKernel_GPUA <16> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);

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
   DCTI_Row__InverseKernel_GPUA <32> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);

   }
 if (strcmp (two,input_buf3) == 0)
{
   DCTII_Row__InverseKernel_GPUA <32> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);

   }
 if (strcmp (three,input_buf3) == 0)
{
   DCTIII_Row__InverseKernel_GPUA <32> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);

   }
 if (strcmp (four,input_buf3) == 0)
{
   DCTIV_Row__InverseKernel_GPUA <32> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);

   }

    plhs[0] = mxGPUCreateMxArrayOnGPU(B);   
    mxGPUDestroyGPUArray(B);
       
       }    
	}

    else if (!(mxIsGPUArray(prhs[0]))){
            
    if (mxGetClassID(prhs[0]) != mxDOUBLE_CLASS) {
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Invalid input to MEX file, input(FIRST ARGUMENT) must be double  precision (double).");
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
	double  * hostA ; // The A matrix
	hostA = (double *)mxGetData(prhs[0]);
    plhs[0] = mxCreateNumericMatrix(numCRows, numCColumns, mxDOUBLE_CLASS, mxREAL);
    double  *pointer =(double*) mxGetPr(plhs[0]);
 
     if (strcmp (one,input_buf3) == 0)
{
      CalculateTransformDCTInverseRowOne(hostA, pointer, numARows, numAColumns, numCRows, numCColumns);  
 } 
     if (strcmp (two,input_buf3) == 0)
{
      CalculateTransformDCTInverseRowTwo(hostA, pointer, numARows, numAColumns, numCRows, numCColumns);  
 } 
     if (strcmp (three,input_buf3) == 0)
{
      CalculateTransformDCTInverseRowThree(hostA, pointer, numARows, numAColumns, numCRows, numCColumns);  
 } 
     if (strcmp (four,input_buf3) == 0)
{
      CalculateTransformDCTInverseRowFour(hostA, pointer, numARows, numAColumns, numCRows, numCColumns);  
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
    double const *d_A;
    double *d_B;
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
    char const * const errMsg = "Invalid input to MEX file, input(FIRST ARGUMENT) must be double  precision (double).";

    if (mxGPUGetClassID(A) != mxDOUBLE_CLASS) {
        mexErrMsgIdAndTxt(errId, errMsg);
    }

    d_A = (double const *)(mxGPUGetDataReadOnly(A));
    mxGPUDestroyGPUArray(A);
    B = mxGPUCreateGPUArray(NrOfDim, (mwSize*) pivot_dimensA, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    d_B = (double *)(mxGPUGetData(B));

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
     
   DSTI_Column_Kernel_GPUA <16> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
   
   
 }  
 if (strcmp (two,input_buf3) == 0)
{
     
   DSTII_Column_Kernel_GPUA <16> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
   
   
 }
 if (strcmp (three,input_buf3) == 0)
{
     
   DSTIII_Column_Kernel_GPUA <16> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
   
   
 }
 if (strcmp (four,input_buf3) == 0)
{
     
   DSTIV_Column_Kernel_GPUA <16> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
   
   
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
     
   DSTI_Column_Kernel_GPUA <32> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
   
   
 }  
 if (strcmp (two,input_buf3) == 0)
{
     
   DSTII_Column_Kernel_GPUA <32> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
   
   
 }
 if (strcmp (three,input_buf3) == 0)
{
     
   DSTIII_Column_Kernel_GPUA <32> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
   
   
 }
 if (strcmp (four,input_buf3) == 0)
{
     
   DSTIV_Column_Kernel_GPUA <32> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
   
   
 }

    plhs[0] = mxGPUCreateMxArrayOnGPU(B);  
    mxGPUDestroyGPUArray(B);
     
       }   
	}

    else if (!(mxIsGPUArray(prhs[0]))){
            
    if (mxGetClassID(prhs[0]) != mxDOUBLE_CLASS) {
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Invalid input to MEX file, input(FIRST ARGUMENT) must be double  precision (double).");
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
	double  * hostA ; // The A matrix
	hostA = (double *)mxGetData(prhs[0]);
    
    plhs[0] = mxCreateNumericMatrix(numCRows, numCColumns, mxDOUBLE_CLASS, mxREAL);
    double  *pointer = (double*)mxGetPr(plhs[0]);
     
     if (strcmp (one,input_buf3) == 0)
    {
     
       CalculateTransformDSTColumnOne(hostA, pointer, numARows, numAColumns, numCRows, numCColumns);
	
    }
     if (strcmp (two,input_buf3) == 0)
    {
     
       CalculateTransformDSTColumnTwo(hostA, pointer, numARows, numAColumns, numCRows, numCColumns);
	
    }
     if (strcmp (three,input_buf3) == 0)
    {
     
       CalculateTransformDSTColumnThree(hostA, pointer, numARows, numAColumns, numCRows, numCColumns);
	
    }
     if (strcmp (four,input_buf3) == 0)
    {
     
       CalculateTransformDSTColumnFour(hostA, pointer, numARows, numAColumns, numCRows, numCColumns);
	
    }

      } 
    } 
  
     if (strcmp (row,input_buf4) == 0)
    {
    
    if (mxIsGPUArray(prhs[0])) {
		
    mxGPUArray const *A;
    mxGPUArray *B;
    double const *d_A;
    double *d_B;
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
    char const * const errMsg = "Invalid input to MEX file, input(FIRST ARGUMENT) must be double  precision (double).";
    
    if (mxGPUGetClassID(A) != mxDOUBLE_CLASS) {
        mexErrMsgIdAndTxt(errId, errMsg);
    }

    d_A = (double const *)(mxGPUGetDataReadOnly(A));
    mxGPUDestroyGPUArray(A);
    B = mxGPUCreateGPUArray(NrOfDim, (mwSize*) pivot_dimensA, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    d_B = (double *)(mxGPUGetData(B));

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
    
   DSTI_Row_Kernel_GPUA <16> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
}
 if (strcmp (two,input_buf3) == 0)
{
    
   DSTII_Row_Kernel_GPUA <16> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
}
 if (strcmp (three,input_buf3) == 0)
{
    
   DSTIII_Row_Kernel_GPUA <16> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
}
 if (strcmp (four,input_buf3) == 0)
{
    
   DSTIV_Row_Kernel_GPUA <16> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
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
    
   DSTI_Row_Kernel_GPUA <32> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
}
 if (strcmp (two,input_buf3) == 0)
{
    
   DSTII_Row_Kernel_GPUA <32> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
}
 if (strcmp (three,input_buf3) == 0)
{
    
   DSTIII_Row_Kernel_GPUA <32> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
}
 if (strcmp (four,input_buf3) == 0)
{
    
   DSTIV_Row_Kernel_GPUA <32> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
}
     
    plhs[0] = mxGPUCreateMxArrayOnGPU(B);   
    mxGPUDestroyGPUArray(B);
    
      }
	}
 
    else if (!(mxIsGPUArray(prhs[0]))){
            
    if (mxGetClassID(prhs[0]) != mxDOUBLE_CLASS) {
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Invalid input to MEX file, input(FIRST ARGUMENT) must be double  precision (double).");
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
	double  * hostA ; // The A matrix
	
 if (numAColumns==1)
 {   
 printf("Attention, this is a column vector, please try Discrete Sine Transform in column wise \n");
 return;
 }
   
    mxInitGPU();
	hostA = (double *)mxGetData(prhs[0]);
    plhs[0] = mxCreateNumericMatrix(numCRows, numCColumns, mxDOUBLE_CLASS, mxREAL);
    double  *pointer = (double*)mxGetPr(plhs[0]);
    
     if (strcmp (one,input_buf3) == 0)
{  
      CalculateTransformDSTRowOne(hostA, pointer, numARows, numAColumns, numCRows, numCColumns);
      
      } 
       if (strcmp (two,input_buf3) == 0)
{  
      CalculateTransformDSTRowTwo(hostA, pointer, numARows, numAColumns, numCRows, numCColumns);
      
      }
       if (strcmp (three,input_buf3) == 0)
{  
      CalculateTransformDSTRowThree(hostA, pointer, numARows, numAColumns, numCRows, numCColumns);
      
      }
       if (strcmp (four,input_buf3) == 0)
{  
      CalculateTransformDSTRowFour(hostA, pointer, numARows, numAColumns, numCRows, numCColumns);
      
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
    double const *d_A;
    double *d_B;
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
    char const * const errMsg = "Invalid input to MEX file, input(FIRST ARGUMENT) must be double  precision (double).";

    if (mxGPUGetClassID(A) != mxDOUBLE_CLASS) {
        mexErrMsgIdAndTxt(errId, errMsg);
    }
    d_A = (double const *)(mxGPUGetDataReadOnly(A));
    mxGPUDestroyGPUArray(A);
    B = mxGPUCreateGPUArray(NrOfDim, (mwSize*) pivot_dimensA, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);    
    d_B = (double *)(mxGPUGetData(B));
    
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
    
   DSTI_Column_Inverse_Kernel_GPUA <16> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
   
   
}
  if (strcmp (two,input_buf3) == 0)
{
    
   DSTII_Column_Inverse_Kernel_GPUA <16> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
   
   
}
 if (strcmp (three,input_buf3) == 0)
{
    
   DSTIII_Column_Inverse_Kernel_GPUA <16> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
   
   
}
 if (strcmp (four,input_buf3) == 0)
{
    
   DSTIV_Column_Inverse_Kernel_GPUA <16> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
   
   
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
    
   DSTI_Column_Inverse_Kernel_GPUA <32> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
   
   
}
  if (strcmp (two,input_buf3) == 0)
{
    
   DSTII_Column_Inverse_Kernel_GPUA <32> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
   
   
}
 if (strcmp (three,input_buf3) == 0)
{
    
   DSTIII_Column_Inverse_Kernel_GPUA <32> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
   
   
}
 if (strcmp (four,input_buf3) == 0)
{
    
   DSTIV_Column_Inverse_Kernel_GPUA <32> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);
   
   
}

    plhs[0] = mxGPUCreateMxArrayOnGPU(B);         
    mxGPUDestroyGPUArray(B);
    
      }     
	}

    else if (!(mxIsGPUArray(prhs[0]))){
            
    if (mxGetClassID(prhs[0]) != mxDOUBLE_CLASS) {
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Invalid input to MEX file, input(FIRST ARGUMENT) must be double  precision (double).");
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
	double  * hostA ; // The A matrix
	hostA = (double *)mxGetData(prhs[0]);
    
    plhs[0] = mxCreateNumericMatrix(numCRows, numCColumns, mxDOUBLE_CLASS, mxREAL);
    double  *pointer = (double*)mxGetPr(plhs[0]);
    
     if (strcmp (one,input_buf3) == 0)
   {
  
       CalculateTransformDSTInverseColumnOne(hostA, pointer, numARows, numAColumns, numCRows, numCColumns);
	
   }
     if (strcmp (two,input_buf3) == 0)
   {
  
       CalculateTransformDSTInverseColumnTwo(hostA, pointer, numARows, numAColumns, numCRows, numCColumns);
	
   }
     if (strcmp (three,input_buf3) == 0)
   {
  
       CalculateTransformDSTInverseColumnThree(hostA, pointer, numARows, numAColumns, numCRows, numCColumns);
	
   }
     if (strcmp (four,input_buf3) == 0)
   {
  
       CalculateTransformDSTInverseColumnFour(hostA, pointer, numARows, numAColumns, numCRows, numCColumns);
	
   }

   } 
} 

    if (strcmp (row,input_buf4) == 0)
    {
    
    if (mxIsGPUArray(prhs[0])) {

    mxGPUArray const *A;
    mxGPUArray *B;
    double const *d_A;
    double *d_B;
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
    char const * const errMsg = "Invalid input to MEX file, input(FIRST ARGUMENT) must be double  precision (double).";

    if (mxGPUGetClassID(A) != mxDOUBLE_CLASS) {
        mexErrMsgIdAndTxt(errId, errMsg);
    }
    
    d_A = (double const *)(mxGPUGetDataReadOnly(A));    
    mxGPUDestroyGPUArray(A);
    B = mxGPUCreateGPUArray(NrOfDim, (mwSize*) pivot_dimensA, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE); 
    d_B = (double *)(mxGPUGetData(B));
    
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
   DSTI_Row__InverseKernel_GPUA <16> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);

   }
 if (strcmp (two,input_buf3) == 0)
{
   DSTII_Row__InverseKernel_GPUA <16> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);

   }
 if (strcmp (three,input_buf3) == 0)
{
   DSTIII_Row__InverseKernel_GPUA <16> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);

   }
 if (strcmp (four,input_buf3) == 0)
{
   DSTIV_Row__InverseKernel_GPUA <16> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);

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
   DSTI_Row__InverseKernel_GPUA <32> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);

   }
 if (strcmp (two,input_buf3) == 0)
{
   DSTII_Row__InverseKernel_GPUA <32> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);

   }
 if (strcmp (three,input_buf3) == 0)
{
   DSTIII_Row__InverseKernel_GPUA <32> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);

   }
 if (strcmp (four,input_buf3) == 0)
{
   DSTIV_Row__InverseKernel_GPUA <32> << <dimGrid, dimBlock >> >(d_A, d_B, numARows, numAColumns, numCRows, numCColumns);

   }

    plhs[0] = mxGPUCreateMxArrayOnGPU(B);    
    mxGPUDestroyGPUArray(B); 
        
      }     
	}
 
    else if (!(mxIsGPUArray(prhs[0]))){
            
    if (mxGetClassID(prhs[0]) != mxDOUBLE_CLASS) {
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Invalid input to MEX file, input(FIRST ARGUMENT) must be double  precision (double).");
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
	double  * hostA ; // The A matrix
	hostA = (double *)mxGetData(prhs[0]);
    
    plhs[0] = mxCreateNumericMatrix(numCRows, numCColumns, mxDOUBLE_CLASS, mxREAL);
    double  *pointer =(double*) mxGetPr(plhs[0]);
    
     if (strcmp (one,input_buf3) == 0)
{
      CalculateTransformDSTInverseRowOne(hostA, pointer, numARows, numAColumns, numCRows, numCColumns);  
 } 
     if (strcmp (two,input_buf3) == 0)
{
      CalculateTransformDSTInverseRowTwo(hostA, pointer, numARows, numAColumns, numCRows, numCColumns);  
 } 
     if (strcmp (three,input_buf3) == 0)
{
      CalculateTransformDSTInverseRowThree(hostA, pointer, numARows, numAColumns, numCRows, numCColumns);  
 } 
     if (strcmp (four,input_buf3) == 0)
{
      CalculateTransformDSTInverseRowFour(hostA, pointer, numARows, numAColumns, numCRows, numCColumns);  
 } 

      }
    }
  } 
}

}
