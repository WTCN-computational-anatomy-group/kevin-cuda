
/*
 * Discrete Cosine/Sine Transform(DCT/DST and IDCT/IDST one to four-all in one)
 * DCT/DST and IDCT/IDST I ---> IV
 * This CUDA code can handle/work with  any type of the input mxArrays, 
 * GPUarray or standard matlab CPU array as input {prhs[0] := mxGPUArray or CPU Array}
 * gpuArray output, B=Discrete_Transform(A, , type of Transform (sine or cosine), type of Transform(direct/inverse), type of DCT/DST or IDCT/IDST, dimensions).
 * Developed at UCL, Institute of Neurology, 12 Queen Square, WC1N 3AR, London
 * Wellcome Trust Centre for Neuroimaging
 * Part of the project SPM(http://www.fil.ion.ucl.ac.uk/spm)
 * Copyright 2018
 * Kevin Bronik
 */

#include "matrix.h"
#include "mex.h"
#include "gpu/mxGPUArray.h"
#include "Discrete_Transform_kernel.cuh"
#include "DCT_I_Column.cuh"
#include "DCT_I_Row.cuh"
#include "DCT_I_Column_Inverse.cuh"
#include "DCT_I_Row_Inverse.cuh"
#include "DCT_II_Row.cuh"
#include "DCT_II_Row_Inverse.cuh"
#include "DCT_II_Column.cuh"
#include "DCT_II_Column_Inverse.cuh"
#include "DCT_III_Row.cuh"
#include "DCT_III_Row_Inverse.cuh"
#include "DCT_III_Column.cuh"
#include "DCT_III_Column_Inverse.cuh"
#include "DCT_IV_Row.cuh"
#include "DCT_IV_Row_Inverse.cuh"
#include "DCT_IV_Column.cuh"
#include "DCT_IV_Column_Inverse.cuh"
#include "DST_I_Column.cuh"
#include "DST_I_Row.cuh"
#include "DST_I_Column_Inverse.cuh"
#include "DST_I_Row_Inverse.cuh"
#include "DST_II_Row.cuh"
#include "DST_II_Row_Inverse.cuh"
#include "DST_II_Column.cuh"
#include "DST_II_Column_Inverse.cuh"
#include "DST_III_Row.cuh"
#include "DST_III_Row_Inverse.cuh"
#include "DST_III_Column.cuh"
#include "DST_III_Column_Inverse.cuh"
#include "DST_IV_Row.cuh"
#include "DST_IV_Row_Inverse.cuh"
#include "DST_IV_Column.cuh"
#include "DST_IV_Column_Inverse.cuh"
//#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#define DEFAULT_DIM 32   
#define 	DELTA(i, j)   ((i==j)?1:0)
//#define TILE_DIM 16
unsigned int TILE_DIM=16;
// DCT
extern "C" void  CalculateTransformDCTColumnOne(float * A, float * C, int numARows,
	int numAColumns, int numCRows, int numCColumns);
    
extern "C" void  CalculateTransformDCTInverseColumnOne(float * A, float * C, int numARows,
	int numAColumns, int numCRows, int numCColumns);
    
extern "C" void  CalculateTransformDCTRowOne(float * A, float * C, int numARows,
	int numAColumns, int numCRows, int numCColumns);
    
extern "C" void  CalculateTransformDCTInverseRowOne(float * A, float * C, int numARows,
	int numAColumns, int numCRows, int numCColumns);    
    
extern "C" void  CalculateTransformDCTRowTwo(float * A, float * C, int numARows,
	int numAColumns, int numCRows, int numCColumns);
    
extern "C" void  CalculateTransformDCTInverseRowTwo(float * A, float * C, int numARows,
	int numAColumns, int numCRows, int numCColumns);
    
extern "C" void  CalculateTransformDCTColumnTwo(float * A, float * C, int numARows,
	int numAColumns, int numCRows, int numCColumns);
    
extern "C" void  CalculateTransformDCTInverseColumnTwo(float * A, float * C, int numARows,
	int numAColumns, int numCRows, int numCColumns);
    
extern "C" void  CalculateTransformDCTColumnThree(float * A, float * C, int numARows,
	int numAColumns, int numCRows, int numCColumns); 
    
extern "C" void  CalculateTransformDCTInverseColumnThree(float * A, float * C, int numARows,
	int numAColumns, int numCRows, int numCColumns);
    
extern "C" void  CalculateTransformDCTRowThree(float * A, float * C, int numARows,
	int numAColumns, int numCRows, int numCColumns);
    
extern "C" void  CalculateTransformDCTInverseRowThree(float * A, float * C, int numARows,
	int numAColumns, int numCRows, int numCColumns);
    
extern "C" void  CalculateTransformDCTColumnFour(float * A, float * C, int numARows,
	int numAColumns, int numCRows, int numCColumns);  
    
extern "C" void  CalculateTransformDCTInverseColumnFour(float * A, float * C, int numARows,
	int numAColumns, int numCRows, int numCColumns);
    
extern "C" void  CalculateTransformDCTRowFour(float * A, float * C, int numARows,
	int numAColumns, int numCRows, int numCColumns);
    
extern "C" void  CalculateTransformDCTInverseRowFour(float * A, float * C, int numARows,
	int numAColumns, int numCRows, int numCColumns); 
    
    // DST
extern "C" void  CalculateTransformDSTColumnOne(float * A, float * C, int numARows,
	int numAColumns, int numCRows, int numCColumns);
    
extern "C" void  CalculateTransformDSTInverseColumnOne(float * A, float * C, int numARows,
	int numAColumns, int numCRows, int numCColumns);
    
extern "C" void  CalculateTransformDSTRowOne(float * A, float * C, int numARows,
	int numAColumns, int numCRows, int numCColumns);
    
extern "C" void  CalculateTransformDSTInverseRowOne(float * A, float * C, int numARows,
	int numAColumns, int numCRows, int numCColumns);    
    
extern "C" void  CalculateTransformDSTRowTwo(float * A, float * C, int numARows,
	int numAColumns, int numCRows, int numCColumns);
    
extern "C" void  CalculateTransformDSTInverseRowTwo(float * A, float * C, int numARows,
	int numAColumns, int numCRows, int numCColumns);
    
extern "C" void  CalculateTransformDSTColumnTwo(float * A, float * C, int numARows,
	int numAColumns, int numCRows, int numCColumns);
    
extern "C" void  CalculateTransformDSTInverseColumnTwo(float * A, float * C, int numARows,
	int numAColumns, int numCRows, int numCColumns);
    
extern "C" void  CalculateTransformDSTColumnThree(float * A, float * C, int numARows,
	int numAColumns, int numCRows, int numCColumns); 
    
extern "C" void  CalculateTransformDSTInverseColumnThree(float * A, float * C, int numARows,
	int numAColumns, int numCRows, int numCColumns);
    
extern "C" void  CalculateTransformDSTRowThree(float * A, float * C, int numARows,
	int numAColumns, int numCRows, int numCColumns);
    
extern "C" void  CalculateTransformDSTInverseRowThree(float * A, float * C, int numARows,
	int numAColumns, int numCRows, int numCColumns);
    
extern "C" void  CalculateTransformDSTColumnFour(float * A, float * C, int numARows,
	int numAColumns, int numCRows, int numCColumns);  
    
extern "C" void  CalculateTransformDSTInverseColumnFour(float * A, float * C, int numARows,
	int numAColumns, int numCRows, int numCColumns);
    
extern "C" void  CalculateTransformDSTRowFour(float * A, float * C, int numARows,
	int numAColumns, int numCRows, int numCColumns);
    
extern "C" void  CalculateTransformDSTInverseRowFour(float * A, float * C, int numARows,
	int numAColumns, int numCRows, int numCColumns); 
    
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

//Discrete_Transform(x, 'cosine', 'direct', 'one' , 'column')
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
    //char const * const errId = "parallel:gpu:mexGPUExample:InvalidInput";
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
    ///////// input_buf0=FIRST, SECOND, THIRD, FOURTH, FIFTH  
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
///////  
 //mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
             //   "Input(FIFTH ARGUMENT) must be 'column' or 'row'.\n.");
    
 if (strcmp (cosine,input_buf1) == 0)
{

 if (strcmp (direct,input_buf2) == 0)
{   
  if (strcmp (column,input_buf4) == 0)
{   
///  input standard GPUarray 
    if (mxIsGPUArray(prhs[0])) {
		//mexErrMsgIdAndTxt(errId, errMsg);
           /* Declare all variables.*/
    mxGPUArray const *A;
    mxGPUArray *B;
    float const *d_A;
    float *d_B;
    int numARows, numAColumns, numCRows,  numCColumns;
    


    /* Initialize the MathWorks GPU API. */
    mxInitGPU();
    cudaError_t error;
    int devID = 0;
    // get number of SMs on this GPU
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
    const mwSize *dims;
   dims=mxGPUGetDimensions(A);
   numARows = (int)dims[0]; /* gets number of rows of A */
   numAColumns = (int)dims[1]; /* gets number of columns of A */
 
 		
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
    
    
    B = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(A),
                            mxGPUGetDimensions(A),
                            mxGPUGetClassID(A),
                            mxGPUGetComplexity(A),
                            MX_GPU_DO_NOT_INITIALIZE);
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

  //	cudaError_t err1 = cudaPeekAtLastError();//To capture last error in function call

	//cudaDeviceSynchronize();//To synchronize the device

      plhs[0] = mxGPUCreateMxArrayOnGPU(B);
      
    mxGPUDestroyGPUArray(A);
    
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

  //	cudaError_t err1 = cudaPeekAtLastError();//To capture last error in function call

	//cudaDeviceSynchronize();//To synchronize the device

      plhs[0] = mxGPUCreateMxArrayOnGPU(B);
      
    mxGPUDestroyGPUArray(A);
    
    mxGPUDestroyGPUArray(B);
    
    
      } 
	}
///  input standard array 

    else if (!(mxIsGPUArray(prhs[0]))){
            
   
   if (mxGetClassID(prhs[0]) != mxSINGLE_CLASS) {
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Invalid input to MEX file, input(FIRST ARGUMENT) must be single precision (float).");
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
    //CalculateTransform(float * A, float * C, int numARows, int numAColumns, int numCRows, int numCColumns)  
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

	//free(hostB);
    } 
    } // Column
///////////////////////////    
     if (strcmp (row,input_buf4) == 0)
{
   ///  input standard GPUarray 
    if (mxIsGPUArray(prhs[0])) {
		
   mxGPUArray const *A;
    

    mxGPUArray *B;
    float const *d_A;
   
    float *d_B;
   
    int numARows, numAColumns, numCRows,  numCColumns;
    
  

    /* Initialize the MathWorks GPU API. */
    mxInitGPU();
    cudaError_t error;
    int devID = 0;
    // get number of SMs on this GPU
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
   const mwSize *dims;
 dims=mxGPUGetDimensions(A);
 numARows = (int)dims[0]; /* gets number of rows of A */
 numAColumns = (int)dims[1]; /* gets number of columns of A */
 
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
    
    
    B = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(A),
                            mxGPUGetDimensions(A),
                            mxGPUGetClassID(A),
                            mxGPUGetComplexity(A),
                            MX_GPU_DO_NOT_INITIALIZE);
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
      
    mxGPUDestroyGPUArray(A);
   
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
      
    mxGPUDestroyGPUArray(A);
   
    mxGPUDestroyGPUArray(B);
    
    
    }
     
	}
///  input standard array 

    else if (!(mxIsGPUArray(prhs[0]))){
            
   if (mxGetClassID(prhs[0]) != mxSINGLE_CLASS) {
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Invalid input to MEX file, input(FIRST ARGUMENT) must be single precision (float).");
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
  
  //one
	//free(hostB);
    } 

    }//row
    ///////////////////////////
}// direct

 if (strcmp (inverse,input_buf2) == 0)
{
    if (strcmp (column,input_buf4) == 0)
{      
///  input standard GPUarray 
    if (mxIsGPUArray(prhs[0])) {
		//mexErrMsgIdAndTxt(errId, errMsg);
           /* Declare all variables.*/
    mxGPUArray const *A;
    mxGPUArray *B;
    float const *d_A;
    float *d_B;
    int numARows, numAColumns, numCRows,  numCColumns;
    


    /* Initialize the MathWorks GPU API. */
    mxInitGPU();
    cudaError_t error;
    int devID = 0;
    // get number of SMs on this GPU
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
    const mwSize *dims;
   dims=mxGPUGetDimensions(A);
   numARows = (int)dims[0]; /* gets number of rows of A */
   numAColumns = (int)dims[1]; /* gets number of columns of A */
 
 		
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
    
    
    B = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(A),
                            mxGPUGetDimensions(A),
                            mxGPUGetClassID(A),
                            mxGPUGetComplexity(A),
                            MX_GPU_DO_NOT_INITIALIZE);
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
  //	cudaError_t err1 = cudaPeekAtLastError();//To capture last error in function call

	//cudaDeviceSynchronize();//To synchronize the device

      plhs[0] = mxGPUCreateMxArrayOnGPU(B);
      
    mxGPUDestroyGPUArray(A);
    
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
  //	cudaError_t err1 = cudaPeekAtLastError();//To capture last error in function call

	//cudaDeviceSynchronize();//To synchronize the device

      plhs[0] = mxGPUCreateMxArrayOnGPU(B);
      
    mxGPUDestroyGPUArray(A);
    
    mxGPUDestroyGPUArray(B);
    
    
    }
     
	}
///  input standard array 

    else if (!(mxIsGPUArray(prhs[0]))){
  
    if (mxGetClassID(prhs[0]) != mxSINGLE_CLASS) {
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Invalid input to MEX file, input(FIRST ARGUMENT) must be single precision (float).");
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
    //CalculateTransform(float * A, float * C, int numARows, int numAColumns, int numCRows, int numCColumns) 
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
	//free(hostB);
    } 
} // Column

    if (strcmp (row,input_buf4) == 0)
{
    
///  input standard GPUarray 
    if (mxIsGPUArray(prhs[0])) {
		//mexErrMsgIdAndTxt(errId, errMsg);
           /* Declare all variables.*/
   mxGPUArray const *A;
   

    mxGPUArray *B;
    float const *d_A;
   
    float *d_B;
   
    int numARows, numAColumns, numCRows,  numCColumns;
    


    /* Initialize the MathWorks GPU API. */
    mxInitGPU();
    cudaError_t error;
    int devID = 0;
    // get number of SMs on this GPU
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
    const mwSize *dims;
    dims=mxGPUGetDimensions(A);
    numARows = (int)dims[0]; /* gets number of rows of A */
    numAColumns = (int)dims[1]; /* gets number of columns of A */
 
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
    
    
    B = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(A),
                            mxGPUGetDimensions(A),
                            mxGPUGetClassID(A),
                            mxGPUGetComplexity(A),
                            MX_GPU_DO_NOT_INITIALIZE);
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
// one
  //	cudaError_t err1 = cudaPeekAtLastError();//To capture last error in function call

	//cudaDeviceSynchronize();//To synchronize the device

      plhs[0] = mxGPUCreateMxArrayOnGPU(B);
      
    mxGPUDestroyGPUArray(A);
    
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
// one
  //	cudaError_t err1 = cudaPeekAtLastError();//To capture last error in function call

	//cudaDeviceSynchronize();//To synchronize the device

      plhs[0] = mxGPUCreateMxArrayOnGPU(B);
      
    mxGPUDestroyGPUArray(A);
    
    mxGPUDestroyGPUArray(B);
    
    
    }
     
	}
///  input standard array 

    else if (!(mxIsGPUArray(prhs[0]))){
            
    if (mxGetClassID(prhs[0]) != mxSINGLE_CLASS) {
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Invalid input to MEX file, input(FIRST ARGUMENT) must be single precision (float).");
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
	//char const * const errId = "parallel:gpu:DCTTWO:InvalidInput";
	//char const * const errMsg = "Invalid input to MEX file, input(FIRST ARGUMENT) must be single precision (float).";

	float  * hostA ; // The A matrix
	
	
	
	hostA = (float *)mxGetData(prhs[0]);
    
    plhs[0] = mxCreateNumericMatrix(numCRows, numCColumns, mxSINGLE_CLASS, mxREAL);
    float  *pointer =(float*) mxGetPr(plhs[0]);
    
    
	//CalculateTransform(hostA, hostB, hostC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
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
//one
   //memcpy(pointer, hostC, numCRows*numCColumns*sizeof(float)); 
   
   // testing
    //    printf("\n  plhs[0]:");
    //    printf("\n");
    //    for (int i = 0; i<numCRows; i++){
	//	for (int j = 0; j<numCColumns; j++){
	//		printf(" %g ", round  (pointer[i * numCColumns + j]));
	//	}
	//	printf("\n");
//	}
	
	//free(hostB);
    }

    }//row

  } // inverse
}
// cosine

//SINE......................................................................................................................................................
 if (strcmp (sine,input_buf1) == 0)
{

 if (strcmp (direct,input_buf2) == 0)
{   
  if (strcmp (column,input_buf4) == 0)
{   
///  input standard GPUarray 
    if (mxIsGPUArray(prhs[0])) {
		//mexErrMsgIdAndTxt(errId, errMsg);
           /* Declare all variables.*/
    mxGPUArray const *A;
    mxGPUArray *B;
    float const *d_A;
    float *d_B;
    int numARows, numAColumns, numCRows,  numCColumns;
    


    /* Initialize the MathWorks GPU API. */
    mxInitGPU();
    cudaError_t error;
    int devID = 0;
    // get number of SMs on this GPU
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
    const mwSize *dims;
   dims=mxGPUGetDimensions(A);
   numARows = (int)dims[0]; /* gets number of rows of A */
   numAColumns = (int)dims[1]; /* gets number of columns of A */
 
 		
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
    
    
    B = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(A),
                            mxGPUGetDimensions(A),
                            mxGPUGetClassID(A),
                            mxGPUGetComplexity(A),
                            MX_GPU_DO_NOT_INITIALIZE);
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

  //	cudaError_t err1 = cudaPeekAtLastError();//To capture last error in function call

	//cudaDeviceSynchronize();//To synchronize the device

      plhs[0] = mxGPUCreateMxArrayOnGPU(B);
      
    mxGPUDestroyGPUArray(A);
    
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

  //	cudaError_t err1 = cudaPeekAtLastError();//To capture last error in function call

	//cudaDeviceSynchronize();//To synchronize the device

      plhs[0] = mxGPUCreateMxArrayOnGPU(B);
      
    mxGPUDestroyGPUArray(A);
    
    mxGPUDestroyGPUArray(B);
    
    
    }
     
	}
///  input standard array 

    else if (!(mxIsGPUArray(prhs[0]))){
            
    if (mxGetClassID(prhs[0]) != mxSINGLE_CLASS) {
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Invalid input to MEX file, input(FIRST ARGUMENT) must be single precision (float).");
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
    //CalculateTransform(float * A, float * C, int numARows, int numAColumns, int numCRows, int numCColumns)  
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

	//free(hostB);
    } 
    } // Column
///////////////////////////    
     if (strcmp (row,input_buf4) == 0)
{
   ///  input standard GPUarray 
    if (mxIsGPUArray(prhs[0])) {
		
   mxGPUArray const *A;
    

    mxGPUArray *B;
    float const *d_A;
   
    float *d_B;
   
    int numARows, numAColumns, numCRows,  numCColumns;
    


    /* Initialize the MathWorks GPU API. */
    mxInitGPU();
    cudaError_t error;
    int devID = 0;
    // get number of SMs on this GPU
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
   const mwSize *dims;
 dims=mxGPUGetDimensions(A);
 numARows = (int)dims[0]; /* gets number of rows of A */
 numAColumns = (int)dims[1]; /* gets number of columns of A */
 
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
    
    
    B = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(A),
                            mxGPUGetDimensions(A),
                            mxGPUGetClassID(A),
                            mxGPUGetComplexity(A),
                            MX_GPU_DO_NOT_INITIALIZE);
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
      
    mxGPUDestroyGPUArray(A);
   
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
      
    mxGPUDestroyGPUArray(A);
   
    mxGPUDestroyGPUArray(B);
    
    }
	}
///  input standard array 

    else if (!(mxIsGPUArray(prhs[0]))){
            
    if (mxGetClassID(prhs[0]) != mxSINGLE_CLASS) {
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Invalid input to MEX file, input(FIRST ARGUMENT) must be single precision (float).");
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
  
  //one
	//free(hostB);
    } 

    }//row
    ///////////////////////////
}// direct

 if (strcmp (inverse,input_buf2) == 0)
{
    if (strcmp (column,input_buf4) == 0)
{      
///  input standard GPUarray 
    if (mxIsGPUArray(prhs[0])) {
		//mexErrMsgIdAndTxt(errId, errMsg);
           /* Declare all variables.*/
    mxGPUArray const *A;
    mxGPUArray *B;
    float const *d_A;
    float *d_B;
    int numARows, numAColumns, numCRows,  numCColumns;
    


    /* Initialize the MathWorks GPU API. */
    mxInitGPU();
    cudaError_t error;
    int devID = 0;
    // get number of SMs on this GPU
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
    const mwSize *dims;
   dims=mxGPUGetDimensions(A);
   numARows = (int)dims[0]; /* gets number of rows of A */
   numAColumns = (int)dims[1]; /* gets number of columns of A */
 
 		
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
    
    
    B = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(A),
                            mxGPUGetDimensions(A),
                            mxGPUGetClassID(A),
                            mxGPUGetComplexity(A),
                            MX_GPU_DO_NOT_INITIALIZE);
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
  //	cudaError_t err1 = cudaPeekAtLastError();//To capture last error in function call

	//cudaDeviceSynchronize();//To synchronize the device

      plhs[0] = mxGPUCreateMxArrayOnGPU(B);
      
    mxGPUDestroyGPUArray(A);
    
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
  //	cudaError_t err1 = cudaPeekAtLastError();//To capture last error in function call

	//cudaDeviceSynchronize();//To synchronize the device

      plhs[0] = mxGPUCreateMxArrayOnGPU(B);
      
    mxGPUDestroyGPUArray(A);
    
    mxGPUDestroyGPUArray(B);
    
    }
     
	}
///  input standard array 

    else if (!(mxIsGPUArray(prhs[0]))){
            
    if (mxGetClassID(prhs[0]) != mxSINGLE_CLASS) {
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Invalid input to MEX file, input(FIRST ARGUMENT) must be single precision (float).");
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
    //CalculateTransform(float * A, float * C, int numARows, int numAColumns, int numCRows, int numCColumns) 
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
	//free(hostB);
    } 
} // Column

    if (strcmp (row,input_buf4) == 0)
{
    
///  input standard GPUarray 
    if (mxIsGPUArray(prhs[0])) {
		//mexErrMsgIdAndTxt(errId, errMsg);
           /* Declare all variables.*/
   mxGPUArray const *A;
   

    mxGPUArray *B;
    float const *d_A;
   
    float *d_B;
   
    int numARows, numAColumns, numCRows,  numCColumns;
    


    /* Initialize the MathWorks GPU API. */
    mxInitGPU();
    cudaError_t error;
    int devID = 0;
    // get number of SMs on this GPU
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
    const mwSize *dims;
    dims=mxGPUGetDimensions(A);
    numARows = (int)dims[0]; /* gets number of rows of A */
    numAColumns = (int)dims[1]; /* gets number of columns of A */
 
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
    
    
    B = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(A),
                            mxGPUGetDimensions(A),
                            mxGPUGetClassID(A),
                            mxGPUGetComplexity(A),
                            MX_GPU_DO_NOT_INITIALIZE);
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
// one
  //	cudaError_t err1 = cudaPeekAtLastError();//To capture last error in function call

	//cudaDeviceSynchronize();//To synchronize the device

      plhs[0] = mxGPUCreateMxArrayOnGPU(B);
      
    mxGPUDestroyGPUArray(A);
    
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
// one
  //	cudaError_t err1 = cudaPeekAtLastError();//To capture last error in function call

	//cudaDeviceSynchronize();//To synchronize the device

      plhs[0] = mxGPUCreateMxArrayOnGPU(B);
      
    mxGPUDestroyGPUArray(A);
    
    mxGPUDestroyGPUArray(B); 
    
    
    } 
     
	}
///  input standard array 

    else if (!(mxIsGPUArray(prhs[0]))){
            
    if (mxGetClassID(prhs[0]) != mxSINGLE_CLASS) {
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Invalid input to MEX file, input(FIRST ARGUMENT) must be single precision (float).");
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
	//char const * const errId = "parallel:gpu:DCTTWO:InvalidInput";
	//char const * const errMsg = "Invalid input to MEX file, input(FIRST ARGUMENT) must be single precision (float).";

	float  * hostA ; // The A matrix
	
	
	
	hostA = (float *)mxGetData(prhs[0]);
    
    plhs[0] = mxCreateNumericMatrix(numCRows, numCColumns, mxSINGLE_CLASS, mxREAL);
    float  *pointer =(float*) mxGetPr(plhs[0]);
    
    
	//CalculateTransform(hostA, hostB, hostC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
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
//one
   //memcpy(pointer, hostC, numCRows*numCColumns*sizeof(float)); 
   
   // testing
    //    printf("\n  plhs[0]:");
    //    printf("\n");
    //    for (int i = 0; i<numCRows; i++){
	//	for (int j = 0; j<numCColumns; j++){
	//		printf(" %g ", round  (pointer[i * numCColumns + j]));
	//	}
	//	printf("\n");
//	}
	
	//free(hostB);
    }

    }//row

  } // inverse
}

    //sine

/////////////END

}
