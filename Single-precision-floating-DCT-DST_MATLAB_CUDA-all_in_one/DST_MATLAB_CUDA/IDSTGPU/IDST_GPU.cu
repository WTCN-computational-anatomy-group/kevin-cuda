/*
 * Inverse Discrete Sine Transform(IDCT one to four)
 * IDST I ---> IV
 * This CUDA code can handle/work with  any type of the input mxArrays, 
 * GPUarray or standard matlab CPU array as input {prhs[0] := mxGPUArray or CPU Array}
 * gpuArray output, B=IDST_GPU(A, type of Inverse DST, dimensions).
 * Developed at UCL, Institute of Neurology, 12 Queen Square, WC1N 3AR, London
 * Wellcome Trust Centre for Neuroimaging
 * Part of the project SPM(http://www.fil.ion.ucl.ac.uk/spm)
 * Copyright 2018
 * Kevin Bronik
 */
#include "matrix.h"
#include "mex.h"
#include "gpu/mxGPUArray.h"
#include "IDST_GPU.cuh"
#define 	DELTA(i, j)   ((i==j)?1:0)
#define DEFAULT_DIM 32 
const double  PI_d = 3.141592653589793238462643383279502884; //pi



__global__ void IDST_GPUA(double const * const A, double const * const B, double * const C,
	int numARows, int numAColumns,
	int numBRows, int numBColumns,
	int numCRows, int numCColumns)
{
	double CValue = 0.0;

	int Row = blockIdx.y*DEFAULT_DIM + threadIdx.y;
	int Col = blockIdx.x*DEFAULT_DIM + threadIdx.x;

	for (int k = 0; k < (DEFAULT_DIM + numAColumns - 1) / DEFAULT_DIM; k++) {

		for (int n = 0; n < DEFAULT_DIM; ++n)
		if ((k*DEFAULT_DIM + n < numAColumns && Row < numARows) && (k*DEFAULT_DIM + n < numBRows && Col < numBColumns))
			CValue += A[Row*numAColumns + k*DEFAULT_DIM + n] * B[(k*DEFAULT_DIM + n)*numBColumns + Col];

	}

	if (Row < numCRows && Col < numCColumns) C[((blockIdx.y * blockDim.y + threadIdx.y)*numCColumns) + (blockIdx.x*blockDim.x) + threadIdx.x] = CValue;

}

__global__ void IDST_Kernel(double  *A, double   *B, double  *C,
	int numARows, int numAColumns,
	int numBRows, int numBColumns,
	int numCRows, int numCColumns)
{
	double CValue = 0.0;

	int Row = blockIdx.y*DEFAULT_DIM + threadIdx.y;
	int Col = blockIdx.x*DEFAULT_DIM + threadIdx.x;

	for (int k = 0; k < (DEFAULT_DIM + numAColumns - 1) / DEFAULT_DIM; k++) {

		for (int n = 0; n < DEFAULT_DIM; ++n)
		if ((k*DEFAULT_DIM + n < numAColumns && Row < numARows) && (k*DEFAULT_DIM + n < numBRows && Col < numBColumns))
			CValue += A[Row*numAColumns + k*DEFAULT_DIM + n] * B[(k*DEFAULT_DIM + n)*numBColumns + Col];

	}

	if (Row < numCRows && Col < numCColumns) C[((blockIdx.y * blockDim.y + threadIdx.y)*numCColumns) + (blockIdx.x*blockDim.x) + threadIdx.x] = CValue;

}


// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void CalculateTransform(double  * A, double  * B, double  * C, int numARows,
	int numAColumns, int numBRows, int numBColumns,
	int numCRows, int numCColumns)
{


	double  * hostA = A; // The A matrix
	double  * hostB = B; // The B matrix
	double * hostC = C; // The output C matrix
	//double * hostComputedC;
	double  * deviceA=0;
	double  * deviceB=0;
	double  * deviceC=0;

	//hostA = (double *)malloc(sizeof(float)*numARows*numAColumns);
	//hostB = (v *)malloc(sizeof(float)*numBRows*numBColumns);

	// Setting numCRows and numCColumns
	numCRows = numARows;
	numCColumns = numBColumns;
	// Allocate GPU buffers for three vectors (two input, one output)    .
	//hostC = (float *)malloc(sizeof(float)*numCRows*numCColumns);
	//hostComputedC = (float *)malloc(sizeof(float)*numCRows*numCColumns);

	
	 cudaMalloc((void **)&deviceA, sizeof(double )*numARows*numAColumns);


	 cudaMalloc((void **)&deviceB, sizeof(double )*numBRows*numBColumns);


	 cudaMalloc((void **)&deviceC, sizeof(double )*numCRows*numCColumns);


	 cudaMemcpy(deviceA, hostA, sizeof(double )*numARows*numAColumns, cudaMemcpyHostToDevice);
	

	 cudaMemcpy(deviceB, hostB, sizeof(double )*numBRows*numBColumns, cudaMemcpyHostToDevice);
	

	dim3 dimBlock(DEFAULT_DIM, DEFAULT_DIM, 1);
	dim3 dimGrid;

	dimGrid.x = (numCColumns + dimBlock.x - 1) / dimBlock.x;
	dimGrid.y = (numCRows + dimBlock.y - 1) / dimBlock.y;
	IDST_Kernel << <dimGrid, dimBlock >> >(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);



	 cudaDeviceSynchronize();//To synchronize the device

	// Copy the results in GPU memory back to the CPU
	 cudaMemcpy(hostC, deviceC, sizeof(double)*numCRows*numCColumns, cudaMemcpyDeviceToHost);

	C = hostC;

	cudaFree(deviceA);
	cudaFree(deviceB);
	cudaFree(deviceC);
    
}

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
    if (nrhs != 3) {
        mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidNumInputs",
                "three input argument required.");
    } 


char row[] = "row";
char column[] = "column";
char one[] = "one";
char two[] = "two";
char three[] = "three";
char four[] = "four";

 char *input_buf1;
 input_buf1 = mxArrayToString(prhs[1]);
char *input_buf2;
 input_buf2 = mxArrayToString(prhs[2]);
      if (!(mxIsChar(prhs[1]))){
        mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Input must be of type string.\n.");
    }
      if (!(mxIsChar(prhs[2]))){
        mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Input must be of type string.\n.");
    }
//  row........................................................................................................ 
if (strcmp (row,input_buf2) == 0)
{

///  input standard GPUarray 
    if (mxIsGPUArray(prhs[0])) {
		//mexErrMsgIdAndTxt(errId, errMsg);
           /* Declare all variables.*/
   mxGPUArray const *A;
    mxGPUArray const *DCOS;

    mxGPUArray *B;
    double const *d_A, *d_DCOS;
   
    double *d_B;
   // mxArray  * hostcos;
    //test
   // double * hostcos, *pointer;
   double  *pointer;
    //int N;
    int numARows, numAColumns,  numDCOSRows,  numDCOSColumns, numCRows,  numCColumns;
    
    char const * const errId = "parallel:gpu:mexGPUExample:InvalidInput";
    char const * const errMsg = "Invalid input to MEX file.";

    /* Initialize the MathWorks GPU API. */
    mxInitGPU();

    /* Throw an error if the input is not a GPU array. */
   // if ((nrhs!=1) || !(mxIsGPUArray(prhs[0]))) {
    //    mexErrMsgIdAndTxt(errId, errMsg);
   // }

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
 
 numDCOSRows=numDCOSColumns=numAColumns;
    numCRows = numARows;
	numCColumns = numDCOSColumns;
 mxArray *COS= mxCreateNumericMatrix(numDCOSRows, numDCOSColumns, mxDOUBLE_CLASS, mxREAL);
pointer = mxGetPr(COS);

//Inverse Discrete Sine Transform in row wise
// DST I
 if (strcmp (one,input_buf1) == 0){
		for (int i = 0; i < numDCOSRows; i++){
			for (int j = 0; j < numDCOSColumns; j++){
				//hostB[i * numBColumns + j] = i + j* numAColumns;
				//hostBinv[i * numBColumns + j] = cosval2x[i + j* numAColumns];
				//hostBinv[i * numBColumns + j] = 1;
				//hostB[i * numBColumns + j] = cosvalx[i + j* numAColumns];
				//hostB[i * numBColumns + j] = cosvalx[i + j* numAColumns];
				//hostB[i * numBColumns + j] = cosvalx[i + j* numAColumns];
				//hostB[i + j* numBColumns] = sin((((j + 1)*3.14*(i + 1)) / (numBColumns + 1)))*sqrt(2.0 / (numBColumns + 1));
				pointer[i + j* numDCOSColumns] = sin((((j + 1)*PI_d*(i + 1)) / (numDCOSColumns + 1)))*sqrt(2.0 / (numDCOSColumns + 1));
				//hostB[i + j* numBColumns] = 1;
					//hostBinv[i + j* numBColumns] = cos((j*3.14*i / (numBColumns - 1)))*sqrt(2.0 / numBColumns)*(1.0 / 2.0);
					//hostB[i + j* numBColumns] = 1;
				

			}
		}
}
// DST II
 if (strcmp (two,input_buf1) == 0){
		for (int i = 0; i < numDCOSRows; i++){
			for (int j = 0; j < numDCOSColumns; j++){
				//hostB[i * numBColumns + j] = i + j* numAColumns;
				//hostBinv[i * numBColumns + j] = cosval2x[i + j* numAColumns];
				//hostBinv[i * numBColumns + j] = 1;
				//hostB[i * numBColumns + j] = cosvalx[i + j* numAColumns];
				//hostB[i * numBColumns + j] = cosvalx[i + j* numAColumns];
				//hostB[i * numBColumns + j] = cosvalx[i + j* numAColumns];
				//hostB[i + j* numBColumns] = sin((((j + 1)*3.14*(i + 1)) / (numBColumns + 1)))*sqrt(2.0 / (numBColumns + 1));
				//hostBinv[i + j* numBColumns] = sin((((j + 1)*3.14*(i + 1)) / (numBColumns + 1)))*sqrt(2.0 / (numBColumns + 1));// DST I
				//hostBinv[i + j* numBColumns] = sin(((j + 1)*3.14*(i + 0.5)) / (numBColumns))*sqrt(2.0 / (numBColumns))*sqrt(1.0 / (1 + DELTA(1, j + 1)));   test
				pointer[i + j* numDCOSColumns] = sin(((j + 1)*PI_d*(i + 0.5)) / (numDCOSColumns))*sqrt(2.0 / (numDCOSColumns))*sqrt(1.0 / (1 + DELTA(numDCOSColumns, j + 1)));
				//hostB[i + j* numBColumns] = 1;
					//hostBinv[i + j* numBColumns] = cos((j*3.14*i / (numBColumns - 1)))*sqrt(2.0 / numBColumns)*(1.0 / 2.0);
					//hostB[i + j* numBColumns] = 1;
				

			}
		}
      } 
// DST III
	if (strcmp (three,input_buf1) == 0){      
		for (int i = 0; i < numDCOSRows; i++){
			for (int j = 0; j < numDCOSColumns; j++){
				//hostB[i * numBColumns + j] = i + j* numAColumns;
				//hostBinv[i * numBColumns + j] = cosval2x[i + j* numAColumns];
				//hostBinv[i * numBColumns + j] = 1;
				//hostB[i * numBColumns + j] = cosvalx[i + j* numAColumns];
				//hostB[i * numBColumns + j] = cosvalx[i + j* numAColumns];
				//hostB[i * numBColumns + j] = cosvalx[i + j* numAColumns];
				//hostB[i + j* numBColumns] = sin((((j + 1)*3.14*(i + 1)) / (numBColumns + 1)))*sqrt(2.0 / (numBColumns + 1));
				//hostBinv[i + j* numBColumns] = sin((((j + 1)*3.14*(i + 1)) / (numBColumns + 1)))*sqrt(2.0 / (numBColumns + 1));// DST I
				//hostBinv[i + j* numBColumns] = sin(((j + 1)*3.14*(i + 0.5)) / (numBColumns))*sqrt(2.0 / (numBColumns))*sqrt(1.0 / (1 + DELTA(1, j + 1)));   test
				pointer[i + j* numDCOSColumns] = sin(((j + 0.5)*PI_d*(i + 1)) / (numDCOSColumns))*sqrt((2.0 - DELTA(i + 1, numDCOSRows)) / (numDCOSColumns));
				//hostB[i + j* numBColumns] = 1;
					//hostBinv[i + j* numBColumns] = cos((j*3.14*i / (numBColumns - 1)))*sqrt(2.0 / numBColumns)*(1.0 / 2.0);
					//hostB[i + j* numBColumns] = 1;
				

			}
		}
  }
 // DST IV
if (strcmp (four,input_buf1) == 0){ 
		for (int i = 0; i < numDCOSRows; i++){
			for (int j = 0; j < numDCOSColumns; j++){
				//hostB[i * numBColumns + j] = i + j* numAColumns;
				//hostBinv[i * numBColumns + j] = cosval2x[i + j* numAColumns];
				//hostBinv[i * numBColumns + j] = 1;
				//hostB[i * numBColumns + j] = cosvalx[i + j* numAColumns];
				//hostB[i * numBColumns + j] = cosvalx[i + j* numAColumns];
				//hostB[i * numBColumns + j] = cosvalx[i + j* numAColumns];
				//hostB[i + j* numBColumns] = sin((((j + 1)*3.14*(i + 1)) / (numBColumns + 1)))*sqrt(2.0 / (numBColumns + 1));
				//hostBinv[i + j* numBColumns] = sin((((j + 1)*3.14*(i + 1)) / (numBColumns + 1)))*sqrt(2.0 / (numBColumns + 1));// DST I
				//hostBinv[i + j* numBColumns] = sin(((j + 1)*3.14*(i + 0.5)) / (numBColumns))*sqrt(2.0 / (numBColumns))*sqrt(1.0 / (1 + DELTA(1, j + 1)));   test
				//hostBinv[i + j* numBColumns] = sin(((j + 0.5)*PI_d*(i + 1)) / (numBColumns))*sqrt((2.0 - DELTA(i + 1, numBRows)) / (numBColumns)); // DST II and III  
				pointer[i + j* numDCOSColumns] = sin(((j + 0.5)*PI_d*(i + 0.5)) / (numDCOSColumns))*sqrt(2.0 / (numDCOSColumns));
				//hostB[i + j* numBColumns] = 1;
					//hostBinv[i + j* numBColumns] = cos((j*3.14*i / (numBColumns - 1)))*sqrt(2.0 / numBColumns)*(1.0 / 2.0);
					//hostB[i + j* numBColumns] = 1;
				

			}
		}
  
  }
      
  
   DCOS=mxGPUCreateFromMxArray(COS);
  //  DCOS=mxGPUCreateFromMxArray(hostcos);

    if (mxGPUGetClassID(A) != mxDOUBLE_CLASS) {
        mexErrMsgIdAndTxt(errId, errMsg);
    }


    d_A = (double const *)(mxGPUGetDataReadOnly(A));
    d_DCOS=(double const *)(mxGPUGetDataReadOnly(DCOS));
    
    B = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(A),
                            mxGPUGetDimensions(A),
                            mxGPUGetClassID(A),
                            mxGPUGetComplexity(A),
                            MX_GPU_DO_NOT_INITIALIZE);
    d_B = (double *)(mxGPUGetData(B));

    
	dim3 dimBlock(DEFAULT_DIM, DEFAULT_DIM, 1);
	dim3 dimGrid;

	dimGrid.x = (numCColumns + dimBlock.x - 1) / dimBlock.x;
	dimGrid.y = (numCRows + dimBlock.y - 1) / dimBlock.y;
   IDST_GPUA<< <dimGrid, dimBlock >> >(d_A, d_DCOS, d_B, numARows, numAColumns, numDCOSRows, numDCOSColumns, numCRows, numCColumns);

   
  //	cudaError_t err1 = cudaPeekAtLastError();//To capture last error in function call

	//cudaDeviceSynchronize();//To synchronize the device

      plhs[0] = mxGPUCreateMxArrayOnGPU(B);
      
    mxGPUDestroyGPUArray(A);
    mxGPUDestroyGPUArray(DCOS);
    mxGPUDestroyGPUArray(B);
     
	}
///  input standard array 

    else if (!(mxIsGPUArray(prhs[0]))){
  	int numARows = (int)mxGetM(prhs[0]); 		// number of rows in the matrix A
	int numAColumns = (int)mxGetN(prhs[0]); 	// number of columns in the matrix A
	int numBRows; 		// number of rows in the matrix B
	int numBColumns; 	// number of columns in the matrix B
	int numCRows;		// number of rows in the matrix C (you have to set this)
	int numCColumns;	// number of columns in the matrix C (you have to set this)
	numBRows = numBColumns = numAColumns;
	numCRows = numARows;
    if (numAColumns==1)
    {   
    printf("Attention, this is a column vector, please try Inverse Discrete Sine Transform in column wise \n");
    return;
    }
	numCColumns = numBColumns;
	//char const * const errId = "parallel:gpu:DCTTWO:InvalidInput";
	//char const * const errMsg = "Invalid input to MEX file.";

	double  * hostA ; // The A matrix
	double  * hostB ; // The B matrix
	


	/* Initialize the MathWorks GPU API. */
	//mxInitGPU();

	/* Throw an error if the input is not a GPU array. */
	//if ((nrhs != 1) || !(mxIsGPUArray(prhs[0]))) {
		//mexErrMsgIdAndTxt(errId, errMsg);
	//}

	//hostA = (double *)malloc(sizeof(double)*numARows*numAColumns);
	//hostAx = (double *)malloc(sizeof(double)*numARows*numAColumns);
	//hostAy = (double *)malloc(sizeof(double)*numARows*numAColumns);
	hostB = (double  *)malloc(sizeof(double)*numBRows*numBColumns);

    
  //const  mxArray *G =prhs[0];
   // if ((nrhs != 1) || (mxIsGPUArray(G))) {
		//mexErrMsgIdAndTxt(errId, errMsg);
    //    G = gather(G);
//	}
	hostA = (double *)mxGetData(prhs[0]);
    // hostA = (double *)mxGetData(G);
    
    //Inverse Discrete Sine Transform in row wise
    
 // DST I
 if (strcmp (one,input_buf1) == 0){   
    		for (int i = 0; i < numBRows; i++){
			for (int j = 0; j < numBColumns; j++){
				//hostB[i * numBColumns + j] = i + j* numAColumns;
				//hostBinv[i * numBColumns + j] = cosval2x[i + j* numAColumns];
				//hostBinv[i * numBColumns + j] = 1;
				//hostB[i * numBColumns + j] = cosvalx[i + j* numAColumns];
				//hostB[i * numBColumns + j] = cosvalx[i + j* numAColumns];
				//hostB[i * numBColumns + j] = cosvalx[i + j* numAColumns];
				//hostB[i + j* numBColumns] = sin((((j + 1)*3.14*(i + 1)) / (numBColumns + 1)))*sqrt(2.0 / (numBColumns + 1));
				hostB[i + j* numBColumns] = sin((((j + 1)*PI_d*(i + 1)) / (numBColumns + 1)))*sqrt(2.0 / (numBColumns + 1));
				//hostB[i + j* numBColumns] = 1;
					//hostBinv[i + j* numBColumns] = cos((j*3.14*i / (numBColumns - 1)))*sqrt(2.0 / numBColumns)*(1.0 / 2.0);
					//hostB[i + j* numBColumns] = 1;
				

			}
		}
  }  
 
// DST II
 if (strcmp (two,input_buf1) == 0){  
    		for (int i = 0; i < numBRows; i++){
			for (int j = 0; j < numBColumns; j++){
				//hostB[i * numBColumns + j] = i + j* numAColumns;
				//hostBinv[i * numBColumns + j] = cosval2x[i + j* numAColumns];
				//hostBinv[i * numBColumns + j] = 1;
				//hostB[i * numBColumns + j] = cosvalx[i + j* numAColumns];
				//hostB[i * numBColumns + j] = cosvalx[i + j* numAColumns];
				//hostB[i * numBColumns + j] = cosvalx[i + j* numAColumns];
				//hostB[i + j* numBColumns] = sin((((j + 1)*3.14*(i + 1)) / (numBColumns + 1)))*sqrt(2.0 / (numBColumns + 1));
				//hostBinv[i + j* numBColumns] = sin((((j + 1)*3.14*(i + 1)) / (numBColumns + 1)))*sqrt(2.0 / (numBColumns + 1));// DST I
				//hostBinv[i + j* numBColumns] = sin(((j + 1)*3.14*(i + 0.5)) / (numBColumns))*sqrt(2.0 / (numBColumns))*sqrt(1.0 / (1 + DELTA(1, j + 1)));   test
				hostB[i + j* numBColumns] = sin(((j + 1)*PI_d*(i + 0.5)) / (numBColumns))*sqrt(2.0 / (numBColumns))*sqrt(1.0 / (1 + DELTA(numBColumns, j + 1)));
				//hostB[i + j* numBColumns] = 1;
					//hostBinv[i + j* numBColumns] = cos((j*3.14*i / (numBColumns - 1)))*sqrt(2.0 / numBColumns)*(1.0 / 2.0);
					//hostB[i + j* numBColumns] = 1;
				

			}
		}
 }       
 // DST III
	if (strcmp (three,input_buf1) == 0){       
    		for (int i = 0; i < numBRows; i++){
			for (int j = 0; j < numBColumns; j++){
				//hostB[i * numBColumns + j] = i + j* numAColumns;
				//hostBinv[i * numBColumns + j] = cosval2x[i + j* numAColumns];
				//hostBinv[i * numBColumns + j] = 1;
				//hostB[i * numBColumns + j] = cosvalx[i + j* numAColumns];
				//hostB[i * numBColumns + j] = cosvalx[i + j* numAColumns];
				//hostB[i * numBColumns + j] = cosvalx[i + j* numAColumns];
				//hostB[i + j* numBColumns] = sin((((j + 1)*3.14*(i + 1)) / (numBColumns + 1)))*sqrt(2.0 / (numBColumns + 1));
				//hostBinv[i + j* numBColumns] = sin((((j + 1)*3.14*(i + 1)) / (numBColumns + 1)))*sqrt(2.0 / (numBColumns + 1));// DST I
				//hostBinv[i + j* numBColumns] = sin(((j + 1)*3.14*(i + 0.5)) / (numBColumns))*sqrt(2.0 / (numBColumns))*sqrt(1.0 / (1 + DELTA(1, j + 1)));   test
				hostB[i + j* numBColumns] = sin(((j + 0.5)*PI_d*(i + 1)) / (numBColumns))*sqrt((2.0 - DELTA(i + 1, numBRows)) / (numBColumns));
				//hostB[i + j* numBColumns] = 1;
					//hostBinv[i + j* numBColumns] = cos((j*3.14*i / (numBColumns - 1)))*sqrt(2.0 / numBColumns)*(1.0 / 2.0);
					//hostB[i + j* numBColumns] = 1;
				

			}
		}
   }
   // DST IV
if (strcmp (four,input_buf1) == 0){
    		for (int i = 0; i < numBRows; i++){
			for (int j = 0; j < numBColumns; j++){
				//hostB[i * numBColumns + j] = i + j* numAColumns;
				//hostBinv[i * numBColumns + j] = cosval2x[i + j* numAColumns];
				//hostBinv[i * numBColumns + j] = 1;
				//hostB[i * numBColumns + j] = cosvalx[i + j* numAColumns];
				//hostB[i * numBColumns + j] = cosvalx[i + j* numAColumns];
				//hostB[i * numBColumns + j] = cosvalx[i + j* numAColumns];
				//hostB[i + j* numBColumns] = sin((((j + 1)*3.14*(i + 1)) / (numBColumns + 1)))*sqrt(2.0 / (numBColumns + 1));
				//hostBinv[i + j* numBColumns] = sin((((j + 1)*3.14*(i + 1)) / (numBColumns + 1)))*sqrt(2.0 / (numBColumns + 1));// DST I
				//hostBinv[i + j* numBColumns] = sin(((j + 1)*3.14*(i + 0.5)) / (numBColumns))*sqrt(2.0 / (numBColumns))*sqrt(1.0 / (1 + DELTA(1, j + 1)));   test
				//hostBinv[i + j* numBColumns] = sin(((j + 0.5)*PI_d*(i + 1)) / (numBColumns))*sqrt((2.0 - DELTA(i + 1, numBRows)) / (numBColumns)); // DST II and III  
				hostB[i + j* numBColumns] = sin(((j + 0.5)*PI_d*(i + 0.5)) / (numBColumns))*sqrt(2.0 / (numBColumns));
				//hostB[i + j* numBColumns] = 1;
					//hostBinv[i + j* numBColumns] = cos((j*3.14*i / (numBColumns - 1)))*sqrt(2.0 / numBColumns)*(1.0 / 2.0);
					//hostB[i + j* numBColumns] = 1;
				

			}
		}
        
  }      
        

	//plhs[0] = mxCreateNumericMatrix(numARows, numBColumns, mxDOUBLE_CLASS, mxREAL);

	//hostC = (double*)mxGetData(plhs[0]);
    plhs[0] = mxCreateNumericMatrix(numCRows, numCColumns, mxDOUBLE_CLASS, mxREAL);
    double  *pointer = mxGetPr(plhs[0]);
    
    
	//CalculateTransform(hostA, hostB, hostC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns); 
      CalculateTransform(hostA, hostB, pointer, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);  

   //memcpy(pointer, hostC, numCRows*numCColumns*sizeof(double)); 
   
   // testing
    //    printf("\n  plhs[0]:");
    //    printf("\n");
    //    for (int i = 0; i<numCRows; i++){
	//	for (int j = 0; j<numCColumns; j++){
	//		printf(" %g ", round  (pointer[i * numCColumns + j]));
	//	}
	//	printf("\n");
//	}
	
	free(hostB);
    }


mxFree(input_buf1);
mxFree(input_buf2);   

    }


//  column........................................................................................................

if (strcmp (column,input_buf2) == 0)
{
///  input standard GPUarray 
    if (mxIsGPUArray(prhs[0])) {
		//mexErrMsgIdAndTxt(errId, errMsg);
           /* Declare all variables.*/
   mxGPUArray const *A;
    mxGPUArray const *DCOS;

    mxGPUArray *B;
    double const *d_A, *d_DCOS;
   
    double *d_B;
   // mxArray  * hostcos;
    //test
   // double * hostcos, *pointer;
   double  *pointer;
    //int N;
    int numARows, numAColumns,  numDCOSRows,  numDCOSColumns, numCRows,  numCColumns;
    
    char const * const errId = "parallel:gpu:mexGPUExample:InvalidInput";
    char const * const errMsg = "Invalid input to MEX file.";

    /* Initialize the MathWorks GPU API. */
    mxInitGPU();

   
   // if ((nrhs!=1)) {
     //   mexErrMsgIdAndTxt(errId, errMsg);
   // }

    A = mxGPUCreateFromMxArray(prhs[0]);
const mwSize *dims;
 dims=mxGPUGetDimensions(A);
 numARows = (int)dims[0]; /* gets number of rows of A */
 numAColumns = (int)dims[1]; /* gets number of columns of A */
 
  		numDCOSRows=numDCOSColumns = numARows;
		numCRows = numARows;

		numCColumns = numAColumns;
 
   if (numARows==1)
 {   
 printf("Attention, this is a row vector, please try Inverse Discrete Sine Transform in row wise \n");
 return;
 }
        
        
 
// 		numDCOSRows=numDCOSColumns = numARows;
// 		numCRows = numARows;
// 
// 		numCColumns = numAColumns;
 
 
 
//  numDCOSRows=numDCOSColumns=numAColumns;
//     numCRows = numARows;
// 	numCColumns = numDCOSColumns;
 mxArray *COS= mxCreateNumericMatrix(numDCOSRows, numDCOSColumns, mxDOUBLE_CLASS, mxREAL);
pointer = mxGetPr(COS);
// DST I
 if (strcmp (one,input_buf1) == 0){ 
		for (int i = 0; i < numDCOSRows; i++){
			for (int j = 0; j < numDCOSColumns; j++){
				//hostB[i * numBColumns + j] = i + j* numAColumns;
				//hostBinv[i * numBColumns + j] = cosval2x[i + j* numAColumns];
				//hostBinv[i * numBColumns + j] = 1;
				//hostBinv[i + j* numBColumns] = sin((((j + 1)*3.14*(i + 1)) / (numBColumns + 1)))*sqrt(2.0 / (numBColumns + 1));
				pointer[i* numDCOSColumns + j] = sin((((j + 1)*PI_d*(i + 1)) / (numDCOSColumns + 1)))*sqrt(2.0 / (numDCOSColumns + 1));
				//hostB[i + j* numBColumns] = 1;
				
					//hostBinvL[i* numBColumns + j] = cos(((2 * i + 1) / (2.0 * numBColumns))*3.14*j)*sqrt(1.0 / numBColumns);
				

			}
		}

}
// DST II
 if (strcmp (two,input_buf1) == 0){
		for (int i = 0; i < numDCOSRows; i++){
			for (int j = 0; j < numDCOSColumns; j++){
				//hostB[i * numBColumns + j] = i + j* numAColumns;
				//hostBinv[i * numBColumns + j] = cosval2x[i + j* numAColumns];
				//hostBinv[i * numBColumns + j] = 1;
				//hostBinv[i + j* numBColumns] = sin((((j + 1)*3.14*(i + 1)) / (numBColumns + 1)))*sqrt(2.0 / (numBColumns + 1));
				//hostBinvL[i* numBColumns + j] = sin((((j + 1)*3.14*(i + 1)) / (numBColumns + 1)))*sqrt(2.0 / (numBColumns + 1));//DST I  Column
				//hostBinv[i + j* numBColumns] = sin(((j + 1)*3.14*(i + 0.5)) / (numBColumns))*sqrt(2.0 / (numBColumns))*sqrt(1.0 / (1 + DELTA(numBColumns, j + 1)));
				pointer[i* numDCOSColumns + j] = sin(((j + 1)*PI_d*(i + 0.5)) / (numDCOSColumns))*sqrt(2.0 / (numDCOSColumns))*sqrt(1.0 / (1 + DELTA(numDCOSColumns, j + 1)));
				//hostB[i + j* numBColumns] = 1;
				
					//hostBinvL[i* numBColumns + j] = cos(((2 * i + 1) / (2.0 * numBColumns))*3.14*j)*sqrt(1.0 / numBColumns);
				

			}
		}
}
// DST III
	if (strcmp (three,input_buf1) == 0){
		for (int i = 0; i < numDCOSRows; i++){
			for (int j = 0; j < numDCOSColumns; j++){
				//hostB[i * numBColumns + j] = i + j* numAColumns;
				//hostBinv[i * numBColumns + j] = cosval2x[i + j* numAColumns];
				//hostBinv[i * numBColumns + j] = 1;
				//hostBinv[i + j* numBColumns] = sin((((j + 1)*3.14*(i + 1)) / (numBColumns + 1)))*sqrt(2.0 / (numBColumns + 1));
				//hostBinvL[i* numBColumns + j] = sin((((j + 1)*3.14*(i + 1)) / (numBColumns + 1)))*sqrt(2.0 / (numBColumns + 1));//DST I  Column
				//hostBinv[i + j* numBColumns] = sin(((j + 1)*3.14*(i + 0.5)) / (numBColumns))*sqrt(2.0 / (numBColumns))*sqrt(1.0 / (1 + DELTA(numBColumns, j + 1)));
				pointer[i* numDCOSColumns + j] = sin(((j + 0.5)*PI_d*(i + 1)) / (numDCOSColumns))*sqrt((2.0 - DELTA(i + 1, numDCOSRows)) / (numDCOSColumns));
				//hostB[i + j* numBColumns] = 1;
				
					//hostBinvL[i* numBColumns + j] = cos(((2 * i + 1) / (2.0 * numBColumns))*3.14*j)*sqrt(1.0 / numBColumns);
				

			}
		}

}

// DST IV
if (strcmp (four,input_buf1) == 0){
		for (int i = 0; i < numDCOSRows; i++){
			for (int j = 0; j < numDCOSColumns; j++){
				//hostB[i * numBColumns + j] = i + j* numAColumns;
				//hostBinv[i * numBColumns + j] = cosval2x[i + j* numAColumns];
				//hostBinv[i * numBColumns + j] = 1;
				//hostBinv[i + j* numBColumns] = sin((((j + 1)*3.14*(i + 1)) / (numBColumns + 1)))*sqrt(2.0 / (numBColumns + 1));
				//hostBinvL[i* numBColumns + j] = sin((((j + 1)*3.14*(i + 1)) / (numBColumns + 1)))*sqrt(2.0 / (numBColumns + 1));//DST I  Column
				//hostBinv[i + j* numBColumns] = sin(((j + 1)*3.14*(i + 0.5)) / (numBColumns))*sqrt(2.0 / (numBColumns))*sqrt(1.0 / (1 + DELTA(numBColumns, j + 1)));
				//hostBinvL[i* numBColumns + j] = sin(((j + 0.5)*PI_d*(i + 1)) / (numBColumns))*sqrt((2.0 - DELTA(i + 1, numBRows)) / (numBColumns));  //DST II and III  Column
				pointer[i* numDCOSColumns + j] = sin(((j + 0.5)*PI_d*(i + 0.5)) / (numDCOSColumns))*sqrt(2.0 / (numDCOSColumns));
				//hostB[i + j* numBColumns] = 1;
				
					//hostBinvL[i* numBColumns + j] = cos(((2 * i + 1) / (2.0 * numBColumns))*3.14*j)*sqrt(1.0 / numBColumns);
				

			}
		}

}

  
   DCOS=mxGPUCreateFromMxArray(COS);
  //  DCOS=mxGPUCreateFromMxArray(hostcos);

    if (mxGPUGetClassID(A) != mxDOUBLE_CLASS) {
        mexErrMsgIdAndTxt(errId, errMsg);
    }


    d_A = (double const *)(mxGPUGetDataReadOnly(A));
    d_DCOS=(double const *)(mxGPUGetDataReadOnly(DCOS));
    
    B = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(A),
                            mxGPUGetDimensions(A),
                            mxGPUGetClassID(A),
                            mxGPUGetComplexity(A),
                            MX_GPU_DO_NOT_INITIALIZE);
    d_B = (double *)(mxGPUGetData(B));

    
	dim3 dimBlock(DEFAULT_DIM, DEFAULT_DIM, 1);
	dim3 dimGrid;

	dimGrid.x = (numCColumns + dimBlock.x - 1) / dimBlock.x;
	dimGrid.y = (numCRows + dimBlock.y - 1) / dimBlock.y;
    //(hostL, hostA, hostC,  numBRows, numBColumns, numARows, numAColumns, numCRows, numCColumns);
   //DCTII_Column_Kernel_GPUA<< <dimGrid, dimBlock >> >(d_A, d_DCOS, d_B, numARows, numAColumns, numDCOSRows, numDCOSColumns, numCRows, numCColumns);
   IDST_GPUA<< <dimGrid, dimBlock >> >(d_DCOS, d_A, d_B, numDCOSRows, numDCOSColumns, numARows, numAColumns, numCRows, numCColumns);
   
  //	cudaError_t err1 = cudaPeekAtLastError();//To capture last error in function call

	//cudaDeviceSynchronize();//To synchronize the device

      plhs[0] = mxGPUCreateMxArrayOnGPU(B);
      
    mxGPUDestroyGPUArray(A);
    mxGPUDestroyGPUArray(DCOS);
    mxGPUDestroyGPUArray(B);
     
	}
///  input standard array 

    else if (!(mxIsGPUArray(prhs[0]))){
  	int numARows = (int)mxGetM(prhs[0]); 		// number of rows in the matrix A
	int numAColumns = (int)mxGetN(prhs[0]); 	// number of columns in the matrix A
	int numBRows; 		// number of rows in the matrix B
	int numBColumns; 	// number of columns in the matrix B
	int numCRows;		// number of rows in the matrix C (you have to set this)
	int numCColumns;	// number of columns in the matrix C (you have to set this)
    	numBRows = numBColumns = numARows;
		numCRows = numARows;

		numCColumns = numAColumns;
    
  if (numARows==1)
 {   
 printf("Attention, this is a row vector, please try Inverse Discrete Sine Transform in row wise \n");
 return;
 }
    
// 	numBRows = numBColumns = numAColumns;
// 	numCRows = numARows;
// 
// 	numCColumns = numBColumns;
	//char const * const errId = "parallel:gpu:DCTTWO:InvalidInput";
	//char const * const errMsg = "Invalid input to MEX file.";

	double  * hostA ; // The A matrix
	double  * hostB ; // The B matrix
	


	/* Initialize the MathWorks GPU API. */
	//mxInitGPU();

	/* Throw an error if the input is not a GPU array. */
	//if ((nrhs != 1) || !(mxIsGPUArray(prhs[0]))) {
		//mexErrMsgIdAndTxt(errId, errMsg);
	//}

	//hostA = (double *)malloc(sizeof(double)*numARows*numAColumns);
	//hostAx = (double *)malloc(sizeof(double)*numARows*numAColumns);
	//hostAy = (double *)malloc(sizeof(double)*numARows*numAColumns);
	hostB = (double  *)malloc(sizeof(double)*numBRows*numBColumns);

    
  //const  mxArray *G =prhs[0];
   // if ((nrhs != 1) || (mxIsGPUArray(G))) {
		//mexErrMsgIdAndTxt(errId, errMsg);
    //    G = gather(G);
//	}
	hostA = (double *)mxGetData(prhs[0]);
    // hostA = (double *)mxGetData(G);
	//Inverse Discrete Sine Transform in Columns wise
    
 // DST I
 if (strcmp (one,input_buf1) == 0){   
    		for (int i = 0; i < numBRows; i++){
			for (int j = 0; j < numBColumns; j++){
				//hostB[i * numBColumns + j] = i + j* numAColumns;
				//hostBinv[i * numBColumns + j] = cosval2x[i + j* numAColumns];
				//hostBinv[i * numBColumns + j] = 1;
				//hostBinv[i + j* numBColumns] = sin((((j + 1)*3.14*(i + 1)) / (numBColumns + 1)))*sqrt(2.0 / (numBColumns + 1));
				hostB[i* numBColumns + j] = sin((((j + 1)*PI_d*(i + 1)) / (numBColumns + 1)))*sqrt(2.0 / (numBColumns + 1));
				//hostB[i + j* numBColumns] = 1;
				
					//hostBinvL[i* numBColumns + j] = cos(((2 * i + 1) / (2.0 * numBColumns))*3.14*j)*sqrt(1.0 / numBColumns);
				

			}
		}
 }
 // DST II
 if (strcmp (two,input_buf1) == 0){
    		for (int i = 0; i < numBRows; i++){
			for (int j = 0; j < numBColumns; j++){
				//hostB[i * numBColumns + j] = i + j* numAColumns;
				//hostBinv[i * numBColumns + j] = cosval2x[i + j* numAColumns];
				//hostBinv[i * numBColumns + j] = 1;
				//hostBinv[i + j* numBColumns] = sin((((j + 1)*3.14*(i + 1)) / (numBColumns + 1)))*sqrt(2.0 / (numBColumns + 1));
				//hostBinvL[i* numBColumns + j] = sin((((j + 1)*3.14*(i + 1)) / (numBColumns + 1)))*sqrt(2.0 / (numBColumns + 1));//DST I  Column
				//hostBinv[i + j* numBColumns] = sin(((j + 1)*3.14*(i + 0.5)) / (numBColumns))*sqrt(2.0 / (numBColumns))*sqrt(1.0 / (1 + DELTA(numBColumns, j + 1)));
				hostB[i* numBColumns + j] = sin(((j + 1)*PI_d*(i + 0.5)) / (numBColumns))*sqrt(2.0 / (numBColumns))*sqrt(1.0 / (1 + DELTA(numBColumns, j + 1)));
				//hostB[i + j* numBColumns] = 1;
				
					//hostBinvL[i* numBColumns + j] = cos(((2 * i + 1) / (2.0 * numBColumns))*3.14*j)*sqrt(1.0 / numBColumns);
				

			}
		}
 
 }
 // DST III
	if (strcmp (three,input_buf1) == 0){
    		for (int i = 0; i < numBRows; i++){
			for (int j = 0; j < numBColumns; j++){
				//hostB[i * numBColumns + j] = i + j* numAColumns;
				//hostBinv[i * numBColumns + j] = cosval2x[i + j* numAColumns];
				//hostBinv[i * numBColumns + j] = 1;
				//hostBinv[i + j* numBColumns] = sin((((j + 1)*3.14*(i + 1)) / (numBColumns + 1)))*sqrt(2.0 / (numBColumns + 1));
				//hostBinvL[i* numBColumns + j] = sin((((j + 1)*3.14*(i + 1)) / (numBColumns + 1)))*sqrt(2.0 / (numBColumns + 1));//DST I  Column
				//hostBinv[i + j* numBColumns] = sin(((j + 1)*3.14*(i + 0.5)) / (numBColumns))*sqrt(2.0 / (numBColumns))*sqrt(1.0 / (1 + DELTA(numBColumns, j + 1)));
				hostB[i* numBColumns + j] = sin(((j + 0.5)*PI_d*(i + 1)) / (numBColumns))*sqrt((2.0 - DELTA(i + 1, numBRows)) / (numBColumns));
				//hostB[i + j* numBColumns] = 1;
				
					//hostBinvL[i* numBColumns + j] = cos(((2 * i + 1) / (2.0 * numBColumns))*3.14*j)*sqrt(1.0 / numBColumns);
				

			}
		}
    
}
 // DST IV
if (strcmp (four,input_buf1) == 0){
    		for (int i = 0; i < numBRows; i++){
			for (int j = 0; j < numBColumns; j++){
				//hostB[i * numBColumns + j] = i + j* numAColumns;
				//hostBinv[i * numBColumns + j] = cosval2x[i + j* numAColumns];
				//hostBinv[i * numBColumns + j] = 1;
				//hostBinv[i + j* numBColumns] = sin((((j + 1)*3.14*(i + 1)) / (numBColumns + 1)))*sqrt(2.0 / (numBColumns + 1));
				//hostBinvL[i* numBColumns + j] = sin((((j + 1)*3.14*(i + 1)) / (numBColumns + 1)))*sqrt(2.0 / (numBColumns + 1));//DST I  Column
				//hostBinv[i + j* numBColumns] = sin(((j + 1)*3.14*(i + 0.5)) / (numBColumns))*sqrt(2.0 / (numBColumns))*sqrt(1.0 / (1 + DELTA(numBColumns, j + 1)));
				//hostBinvL[i* numBColumns + j] = sin(((j + 0.5)*PI_d*(i + 1)) / (numBColumns))*sqrt((2.0 - DELTA(i + 1, numBRows)) / (numBColumns));  //DST II and III  Column
				hostB[i* numBColumns + j] = sin(((j + 0.5)*PI_d*(i + 0.5)) / (numBColumns))*sqrt(2.0 / (numBColumns));
				//hostB[i + j* numBColumns] = 1;
				
					//hostBinvL[i* numBColumns + j] = cos(((2 * i + 1) / (2.0 * numBColumns))*3.14*j)*sqrt(1.0 / numBColumns);
				

			}
		}
 }
	//plhs[0] = mxCreateNumericMatrix(numARows, numBColumns, mxDOUBLE_CLASS, mxREAL);

	//hostC = (double*)mxGetData(plhs[0]);
    plhs[0] = mxCreateNumericMatrix(numCRows, numCColumns, mxDOUBLE_CLASS, mxREAL);
    double  *pointer = mxGetPr(plhs[0]);
    
   // (hostL, hostA, hostC,  numBRows, numBColumns, numARows, numAColumns, numCRows, numCColumns);
	//CalculateTransform(hostA, hostB, hostC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns); 
     // CalculateTransform(hostA, hostB, pointer, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);  
       CalculateTransform( hostB, hostA, pointer, numBRows, numBColumns, numARows, numAColumns, numCRows, numCColumns);
   //memcpy(pointer, hostC, numCRows*numCColumns*sizeof(double)); 

	
	free(hostB);
    }



 mxFree(input_buf1);
 mxFree(input_buf2);
    
	}
}
