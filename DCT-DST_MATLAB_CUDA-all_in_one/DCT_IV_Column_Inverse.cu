/*
 * Inverse Discrete Cosine Transform in Column wise (DCT four)
 * DCT_IV_Column_Inverse
 * This CUDA code can handle/work with  any type of the input mxArrays, 
 * GPUarray or standard matlab CPU array as input {prhs[0] := mxGPUArray or CPU Array}
 * gpuArray output, B=DCT_IV_Column_Inverse(A)=mexFunction(A).
 * Developed at UCL, Institute of Neurology, 12 Queen Square,  WC1N 3AR, London
 * Wellcome Trust Centre for Neuroimaging
 * Part of the project SPM(http://www.fil.ion.ucl.ac.uk/spm)
 * Copyright 2018
 * Kevin Bronik
 */
#include "matrix.h"
#include "DCT_IV_Column_Inverse.cuh"
#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cuda.h>
#include <cuda_runtime.h>
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}
#define TILE_DIM 16

#define DEFAULT_DIM 32                     // Tile dimension 
#define 	DELTA(i, j)   ((i==j)?1:0)

//const double  PI_d = 3.141592653589793238462643383279502884; //pi


__global__ void DCTIV_Column_Inverse_Kernel(double  *A, double  *C,
	int numARows, int numAColumns,
	int numCRows, int numCColumns)
{
	double CValue = 0.0;
	const double  PI_d = 3.141592653589793238462643383279502884; //pi
	int Row = blockIdx.y*TILE_DIM + threadIdx.y;
	int Col = blockIdx.x*TILE_DIM + threadIdx.x;

	__shared__ double As[TILE_DIM][TILE_DIM];
	__shared__ double Bs[TILE_DIM][TILE_DIM];

	for (int k = 0; k < (TILE_DIM + numARows - 1) / TILE_DIM; k++) {

		if (k*TILE_DIM + threadIdx.x < numARows && Row < numARows)	{ As[threadIdx.y][threadIdx.x] = cos(((2 * Row + 1)*PI_d*(2 * (threadIdx.x + (k*TILE_DIM)) + 1) / (4.0 * numARows)))*sqrt(2.0 / numARows); }
		
		else											{ As[threadIdx.y][threadIdx.x] = 0.0; }

		if (k*TILE_DIM + threadIdx.y < numARows && Col < numAColumns){ Bs[threadIdx.y][threadIdx.x] = A[(k*TILE_DIM + threadIdx.y)*numAColumns + Col]; }
		else								{ Bs[threadIdx.y][threadIdx.x] = 0.0; }

		__syncthreads();

		for (int n = 0; n < TILE_DIM; ++n) { CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x]; }

		__syncthreads();
	}

	if (Row < numCRows && Col < numCColumns) { C[((blockIdx.y * blockDim.y + threadIdx.y)*numCColumns) + (blockIdx.x*blockDim.x) + threadIdx.x] = CValue; }

}


// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
extern "C" void  CalculateTransformDCTInverseColumnFour(double * A, double * C, int numARows,
	int numAColumns, int numCRows, int numCColumns)
{

	double * hostA = A; // The A matrix
	//double * hostB = B; // The B matrix
	double * hostC = C; // The output C matrix
	//float * hostComputedC;
	double * deviceA;
	//double * deviceB;
	double * deviceC;

	//hostA = (float *)malloc(sizeof(float)*numARows*numAColumns);
	//hostB = (float *)malloc(sizeof(float)*numBRows*numBColumns);




	// Setting numCRows and numCColumns
	numCRows = numARows;
	numCColumns = numAColumns;

	//hostC = (float *)malloc(sizeof(float)*numCRows*numCColumns);
	//hostComputedC = (float *)malloc(sizeof(float)*numCRows*numCColumns);

	// Allocating GPU memory
	gpuErrchk(cudaMalloc((void **)&deviceA, sizeof(double)*numARows*numAColumns));
	//cudaMalloc((void **)&deviceB, sizeof(double)*numBRows*numBColumns);
	gpuErrchk(cudaMalloc((void **)&deviceC, sizeof(double)*numCRows*numCColumns));

	//thrust::device_ptr< double >dev_ptr_A(deviceA);
	//thrust::device_ptr< double >dev_ptr_C(deviceC);

	// Copy memory to the GPU
	gpuErrchk(cudaMemcpy(deviceA, hostA, sizeof(double)*numARows*numAColumns, cudaMemcpyHostToDevice));
	//cudaMemcpy(deviceB, hostB, sizeof(double)*numBRows*numBColumns, cudaMemcpyHostToDevice);

	/////////////////////////////////////////////////////////
	dim3 dimBlock(TILE_DIM, TILE_DIM, 1);
	dim3 dimGrid;

	dimGrid.x = (numCColumns + dimBlock.x - 1) / dimBlock.x;
	dimGrid.y = (numCRows + dimBlock.y - 1) / dimBlock.y;
	DCTIV_Column_Inverse_Kernel << <dimGrid, dimBlock >> >(deviceA, deviceC, numARows, numAColumns, numCRows, numCColumns);
	
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	// Copy the results in GPU memory back to the CPU
	gpuErrchk(cudaMemcpy(hostC, deviceC, sizeof(double)*numCRows*numCColumns, cudaMemcpyDeviceToHost));

	C = hostC;

	//thrust::device_free(dev_ptr_A);
	//thrust::device_free(dev_ptr_C);
	gpuErrchk(cudaFree(deviceA));
	//cudaFree(deviceB);
	gpuErrchk(cudaFree(deviceC));
	return;
    
}
