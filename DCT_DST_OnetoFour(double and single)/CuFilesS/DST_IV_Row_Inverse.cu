/*
 * Inverse Discrete Sine Transform in row wise (DST four)
 * DST_IV_Row_Inverse
 * This CUDA code can handle/work with  any type of the input mxArrays, 
 * GPUarray or standard matlab CPU array as input {prhs[0] := mxGPUArray or CPU Array}
 * gpuArray output, B=DST_IV_Row_Inverse(A)=mexFunction(A).
 * Developed at UCL, Institute of Neurology, 12 Queen Square, WC1N 3AR, London
 * Wellcome Trust Centre for Neuroimaging
 * Part of the project SPM(http://www.fil.ion.ucl.ac.uk/spm)
 * Copyright 2018
 * Kevin Bronik
 */
#include "matrix.h"
#include "DST_IV_Row_Inverse.cuh"
#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include "ERRORCHK.h"
// #define TILE_DIM 16

#define DEFAULT_DIM 32                     // Tile dimension 
#define 	DELTA(i, j)   ((i==j)?1:0)

//const float  PI_d = 3.141592653589793238462643383279502884f; //pi



template <unsigned int TILE_DIM >  __global__ void DSTIV_Row__InverseKernelx(float  *A, float  *C,
	int numARows, int numAColumns,
	int numCRows, int numCColumns)
{
	float CValue = 0.0f;
	const float  PI_d = 3.141592653589793238462643383279502884f; //pi
	int Row = blockIdx.y*TILE_DIM + threadIdx.y;
	int Col = blockIdx.x*TILE_DIM + threadIdx.x;

	__shared__ float As[TILE_DIM][TILE_DIM];
	__shared__ float Bs[TILE_DIM][TILE_DIM];

	for (int k = 0; k < (TILE_DIM + numAColumns - 1) / TILE_DIM; k++) {

		if (k*TILE_DIM + threadIdx.x < numAColumns && Row < numARows)	{ As[threadIdx.y][threadIdx.x] = A[Row*numAColumns + k*TILE_DIM + threadIdx.x]; }
		else													{ As[threadIdx.y][threadIdx.x] = 0.0; }
                                                                           //Bs[threadIdx.y][threadIdx.x] = __cosf(((2 * (threadIdx.y + k*TILE_DIM) + 1)*PI_d*(2 * Col + 1) / (4.0 * numAColumns)))*sqrtf(2.0 / numAColumns);
		if (k*TILE_DIM + threadIdx.y < numAColumns && Col < numAColumns)	{ Bs[threadIdx.y][threadIdx.x] = __sinf((((threadIdx.y + k*TILE_DIM) + 0.5)*PI_d*(Col + 0.5)) / (numAColumns))*sqrtf(2.0 / (numAColumns)); }
		
		else													{ Bs[threadIdx.y][threadIdx.x] = 0.0; }

		__syncthreads();

		for (int n = 0; n < TILE_DIM; ++n) { CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x]; }

		__syncthreads();
	}

	if (Row < numCRows && Col < numCColumns) { C[((blockIdx.y * blockDim.y + threadIdx.y)*numCColumns) + (blockIdx.x*blockDim.x) + threadIdx.x] = CValue; }

}


// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
extern "C" void  CalculateTransformDSTInverseRowFourS(float * A, float * C, int numARows,
	int numAColumns, int numCRows, int numCColumns)
{


	float * hostA = A; // The A matrix
	//float * hostB = B; // The B matrix
	float * hostC = C; // The output C matrix
	//float * hostComputedC;
	float * deviceA;
	//float * deviceB;
	float * deviceC;

	//hostA = (float *)malloc(sizeof(float)*numARows*numAColumns);
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
	
	


	// Setting numCRows and numCColumns
	numCRows = numARows;
	numCColumns = numAColumns;

	//hostC = (float *)malloc(sizeof(float)*numCRows*numCColumns);
	//hostComputedC = (float *)malloc(sizeof(float)*numCRows*numCColumns);

	// Allocating GPU memory
	gpuErrchk(cudaMalloc((void **)&deviceA, sizeof(float)*numARows*numAColumns));
	//cudaMalloc((void **)&deviceB, sizeof(float)*numBRows*numBColumns);
	gpuErrchk(cudaMalloc((void **)&deviceC, sizeof(float)*numCRows*numCColumns));
	
	//thrust::device_ptr< float >dev_ptr_A(deviceA);
	//thrust::device_ptr< float >dev_ptr_C(deviceC);

	// Copy memory to the GPU
	gpuErrchk(cudaMemcpy(deviceA, hostA, sizeof(float)*numARows*numAColumns, cudaMemcpyHostToDevice));
	//cudaMemcpy(deviceB, hostB, sizeof(float)*numBRows*numBColumns, cudaMemcpyHostToDevice);

	/////////////////////////////////////////////////////////
    unsigned int TILE_DIM=16;
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
	DSTIV_Row__InverseKernelx <16> << <dimGrid, dimBlock >> >(deviceA, deviceC, numARows, numAColumns, numCRows, numCColumns);
	//matrixMultiplyShared << <dimGrid, dimBlock >> >(thrust::raw_pointer_cast(&dev_ptr_A[0]), thrust::raw_pointer_cast(&dev_ptr_C[0]), numARows, numAColumns, numCRows, numCColumns);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	// Copy the results in GPU memory back to the CPU
	gpuErrchk(cudaMemcpy(hostC, deviceC, sizeof(float)*numCRows*numCColumns, cudaMemcpyDeviceToHost));

	C = hostC;

	//thrust::device_free(dev_ptr_A);
	//thrust::device_free(dev_ptr_C);
	gpuErrchk(cudaFree(deviceA));
	//cudaFree(deviceB);
	gpuErrchk(cudaFree(deviceC));
	return;
	
	     case 32:
     TILE_DIM= TILEDIM;
	 dimBlock.x=TILE_DIM;
	 dimBlock.y=TILE_DIM;
     dimBlock.z=1;
	

	dimGrid.x = (numCColumns + dimBlock.x - 1) / dimBlock.x;
	dimGrid.y = (numCRows + dimBlock.y - 1) / dimBlock.y;
		DSTIV_Row__InverseKernelx <32> << <dimGrid, dimBlock >> >(deviceA, deviceC, numARows, numAColumns, numCRows, numCColumns);
	//matrixMultiplyShared << <dimGrid, dimBlock >> >(thrust::raw_pointer_cast(&dev_ptr_A[0]), thrust::raw_pointer_cast(&dev_ptr_C[0]), numARows, numAColumns, numCRows, numCColumns);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	// Copy the results in GPU memory back to the CPU
	gpuErrchk(cudaMemcpy(hostC, deviceC, sizeof(float)*numCRows*numCColumns, cudaMemcpyDeviceToHost));

	C = hostC;

	//thrust::device_free(dev_ptr_A);
	//thrust::device_free(dev_ptr_C);
	gpuErrchk(cudaFree(deviceA));
	//cudaFree(deviceB);
	gpuErrchk(cudaFree(deviceC));
	return;
	
	
	}
	
    
}
