
#ifndef _DISCRETE_TRANSFORMS_KERNEL_H_
#define _DISCRETE_TRANSFORMS_KERNEL_H_

#include "matrix.h"
#include "mex.h"
#include "gpu/mxGPUArray.h"
//#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#define DEFAULT_DIM 32   
#define 	DELTA(i, j)   ((i==j)?1:0)
// #define TILE_DIM 16


// cosine
template <unsigned int TILE_DIM >  __global__ void DCTI_Column_Kernel_GPUAS(float const * const A, float * const C,
	int numARows, int numAColumns,
	int numCRows, int numCColumns)
{
	float CValue = 0.0f;
	const float  PI_d = 3.141592653589793238462643383279502884f; //pi
	int Row = blockIdx.y*TILE_DIM + threadIdx.y;
	int Col = blockIdx.x*TILE_DIM + threadIdx.x;

	__shared__ float As[TILE_DIM][TILE_DIM];
	__shared__ float Bs[TILE_DIM][TILE_DIM];

	for (int k = 0; k < (TILE_DIM + numARows - 1) / TILE_DIM; k++) {

		if (k*TILE_DIM + threadIdx.x < numARows && Row < numARows)	{ As[threadIdx.y][threadIdx.x] = __cosf(((threadIdx.x + k*TILE_DIM)*PI_d*Row / (numARows - 1)))*sqrtf(1.0 / (1 + DELTA(Row + 1, 1) + DELTA(Row + 1, numARows)))*sqrtf(1.0 / (1 + DELTA(1, (threadIdx.x + k*TILE_DIM) + 1) + DELTA(numARows, (threadIdx.x + k*TILE_DIM) + 1)))*sqrtf(2.0 / (numARows-1)); }
     
		//As[threadIdx.y][threadIdx.x] = A[Row*ACols + k*TILE_DIM + threadIdx.x];
		else											{ As[threadIdx.y][threadIdx.x] = 0.0; }

		if (k*TILE_DIM + threadIdx.y < numARows && Col < numAColumns){ Bs[threadIdx.y][threadIdx.x] = A[(k*TILE_DIM + threadIdx.y)*numAColumns + Col]; }
		else								{ Bs[threadIdx.y][threadIdx.x] = 0.0; }

		__syncthreads();

		for (int n = 0; n < TILE_DIM; ++n) { CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x]; }

		__syncthreads();
	}

	if (Row < numCRows && Col < numCColumns) { C[((blockIdx.y * blockDim.y + threadIdx.y)*numCColumns) + (blockIdx.x*blockDim.x) + threadIdx.x] = CValue; }

}

template <unsigned int TILE_DIM >  __global__ void DCTI_Column_Inverse_Kernel_GPUAS(float const * const A, float * const C,
	int numARows, int numAColumns,
	int numCRows, int numCColumns)
{
	float CValue = 0.0f;
	const float  PI_d = 3.141592653589793238462643383279502884f; //pi
	int Row = blockIdx.y*TILE_DIM + threadIdx.y;
	int Col = blockIdx.x*TILE_DIM + threadIdx.x;

	__shared__ float As[TILE_DIM][TILE_DIM];
	__shared__ float Bs[TILE_DIM][TILE_DIM];

	for (int k = 0; k < (TILE_DIM + numARows - 1) / TILE_DIM; k++) {

		if (k*TILE_DIM + threadIdx.x < numARows && Row < numARows)	{ As[threadIdx.y][threadIdx.x] = __cosf(((threadIdx.x + k*TILE_DIM)*PI_d*Row / (numARows - 1)))*sqrtf(1.0 / (1 + DELTA(Row + 1, 1) + DELTA(Row + 1, numARows)))*sqrtf(1.0 / (1 + DELTA(1, (threadIdx.x + k*TILE_DIM) + 1) + DELTA(numARows, (threadIdx.x + k*TILE_DIM) + 1)))*sqrtf(2.0 / (numARows-1)); }
		//As[threadIdx.y][threadIdx.x] = A[Row*ACols + k*TILE_DIM + threadIdx.x];
		else											{ As[threadIdx.y][threadIdx.x] = 0.0; }

		if (k*TILE_DIM + threadIdx.y < numARows && Col < numAColumns){ Bs[threadIdx.y][threadIdx.x] = A[(k*TILE_DIM + threadIdx.y)*numAColumns + Col]; }
		else								{ Bs[threadIdx.y][threadIdx.x] = 0.0; }

		__syncthreads();

		for (int n = 0; n < TILE_DIM; ++n) { CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x]; }

		__syncthreads();
	}

	if (Row < numCRows && Col < numCColumns) { C[((blockIdx.y * blockDim.y + threadIdx.y)*numCColumns) + (blockIdx.x*blockDim.x) + threadIdx.x] = CValue; }

}
template <unsigned int TILE_DIM >  __global__ void DCTI_Row_Kernel_GPUAS(float const * const A, float * const C,
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

		if (k*TILE_DIM + threadIdx.y < numAColumns && Col < numAColumns)	{ Bs[threadIdx.y][threadIdx.x] = __cosf(((threadIdx.y + k*TILE_DIM)*PI_d*Col / (numAColumns - 1)))*sqrtf(1.0 / (1 + DELTA(Col + 1, 1) + DELTA(Col + 1, numAColumns)))*sqrtf(1.0 / (1 + DELTA(1, (threadIdx.y + k*TILE_DIM) + 1) + DELTA(numAColumns, (threadIdx.y + k*TILE_DIM) + 1)))*sqrtf(2.0 / (numAColumns-1)); }
		//Bs[threadIdx.y][threadIdx.x] = B[(k*TILE_DIM + threadIdx.y)*BCols + Col];
		else													{ Bs[threadIdx.y][threadIdx.x] = 0.0; }

		__syncthreads();

		for (int n = 0; n < TILE_DIM; ++n) { CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x]; }

		__syncthreads();
	}

	if (Row < numCRows && Col < numCColumns) { C[((blockIdx.y * blockDim.y + threadIdx.y)*numCColumns) + (blockIdx.x*blockDim.x) + threadIdx.x] = CValue; }

}

template <unsigned int TILE_DIM >  __global__ void DCTI_Row__InverseKernel_GPUAS(float const * const A, float * const C,
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

		if (k*TILE_DIM + threadIdx.y < numAColumns && Col < numAColumns)	{ Bs[threadIdx.y][threadIdx.x] = __cosf(((threadIdx.y + k*TILE_DIM)*PI_d*Col / (numAColumns - 1)))*sqrtf(1.0 / (1 + DELTA(Col + 1, 1) + DELTA(Col + 1, numAColumns)))*sqrtf(1.0 / (1 + DELTA(1, (threadIdx.y + k*TILE_DIM) + 1) + DELTA(numAColumns, (threadIdx.y + k*TILE_DIM) + 1)))*sqrtf(2.0 / (numAColumns-1)); }
		//Bs[threadIdx.y][threadIdx.x] = B[(k*TILE_DIM + threadIdx.y)*BCols + Col];
		else													{ Bs[threadIdx.y][threadIdx.x] = 0.0; }

		__syncthreads();

		for (int n = 0; n < TILE_DIM; ++n) { CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x]; }

		__syncthreads();
	}

	if (Row < numCRows && Col < numCColumns) { C[((blockIdx.y * blockDim.y + threadIdx.y)*numCColumns) + (blockIdx.x*blockDim.x) + threadIdx.x] = CValue; }

}
template <unsigned int TILE_DIM >  __global__ void DCTII_Row_Kernel_GPUAS(float const * const A, float * const C,
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

		if (k*TILE_DIM + threadIdx.y < numAColumns && Col < numAColumns)	{ Bs[threadIdx.y][threadIdx.x] = __cosf(((2 * (threadIdx.y + k*TILE_DIM) + 1) / (2.0 * numAColumns))*PI_d*Col)*sqrtf(1.0 / (1 + DELTA(Col + 1, 1)))*sqrtf(2.0 / numAColumns); }
		//Bs[threadIdx.y][threadIdx.x] = B[(k*TILE_DIM + threadIdx.y)*BCols + Col];
		else													{ Bs[threadIdx.y][threadIdx.x] = 0.0; }

		__syncthreads();

		for (int n = 0; n < TILE_DIM; ++n) { CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x]; }

		__syncthreads();
	}

	if (Row < numCRows && Col < numCColumns) { C[((blockIdx.y * blockDim.y + threadIdx.y)*numCColumns) + (blockIdx.x*blockDim.x) + threadIdx.x] = CValue; }

}
template <unsigned int TILE_DIM >  __global__ void DCTII_Row__InverseKernel_GPUAS(float const * const A, float * const C,
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

		if (k*TILE_DIM + threadIdx.y < numAColumns && Col < numAColumns)	{ Bs[threadIdx.y][threadIdx.x] = __cosf(((2 * Col + 1) / (2.0 * numAColumns))*PI_d*(threadIdx.y + k*TILE_DIM))*sqrtf(1.0 / (1 + DELTA(1, (threadIdx.y + k*TILE_DIM) + 1)))*sqrtf(2.0 / numAColumns); }
		//Bs[threadIdx.y][threadIdx.x] = B[(k*TILE_DIM + threadIdx.y)*BCols + Col];
		else													{ Bs[threadIdx.y][threadIdx.x] = 0.0; }

		__syncthreads();

		for (int n = 0; n < TILE_DIM; ++n) { CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x]; }

		__syncthreads();
	}

	if (Row < numCRows && Col < numCColumns) { C[((blockIdx.y * blockDim.y + threadIdx.y)*numCColumns) + (blockIdx.x*blockDim.x) + threadIdx.x] = CValue; }

}
template <unsigned int TILE_DIM >  __global__ void DCTII_Column_Kernel_GPUAS(float const * const A, float * const C,
	int numARows, int numAColumns,
	int numCRows, int numCColumns)
{
	float CValue = 0.0f;
	const float  PI_d = 3.141592653589793238462643383279502884f; //pi
	int Row = blockIdx.y*TILE_DIM + threadIdx.y;
	int Col = blockIdx.x*TILE_DIM + threadIdx.x;

	__shared__ float As[TILE_DIM][TILE_DIM];
	__shared__ float Bs[TILE_DIM][TILE_DIM];

	for (int k = 0; k < (TILE_DIM + numARows - 1) / TILE_DIM; k++) {

		if (k*TILE_DIM + threadIdx.x < numARows && Row < numARows)	{ As[threadIdx.y][threadIdx.x] = __cosf(((2 * (threadIdx.x + k*TILE_DIM) + 1) / (2.0 * numARows))*PI_d*Row)*sqrtf(1.0 / (1 + DELTA(Row + 1, 1)))*sqrtf(2.0 / numARows); }
  	
		else											{ As[threadIdx.y][threadIdx.x] = 0.0; }

		if (k*TILE_DIM + threadIdx.y < numARows && Col < numAColumns){ Bs[threadIdx.y][threadIdx.x] = A[(k*TILE_DIM + threadIdx.y)*numAColumns + Col]; }
		else								{ Bs[threadIdx.y][threadIdx.x] = 0.0; }

		__syncthreads();

		for (int n = 0; n < TILE_DIM; ++n) { CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x]; }

		__syncthreads();
	}

	if (Row < numCRows && Col < numCColumns) { C[((blockIdx.y * blockDim.y + threadIdx.y)*numCColumns) + (blockIdx.x*blockDim.x) + threadIdx.x] = CValue; }

}
template <unsigned int TILE_DIM >  __global__ void DCTII_Column_Inverse_Kernel_GPUAS(float const * const A, float * const C,
	int numARows, int numAColumns,
	int numCRows, int numCColumns)
{
	float CValue = 0.0f;
	const float  PI_d = 3.141592653589793238462643383279502884f; //pi
	int Row = blockIdx.y*TILE_DIM + threadIdx.y;
	int Col = blockIdx.x*TILE_DIM + threadIdx.x;

	__shared__ float As[TILE_DIM][TILE_DIM];
	__shared__ float Bs[TILE_DIM][TILE_DIM];

	for (int k = 0; k < (TILE_DIM + numARows - 1) / TILE_DIM; k++) {

		if (k*TILE_DIM + threadIdx.x < numARows && Row < numARows)	{ As[threadIdx.y][threadIdx.x] = __cosf(((2 * Row + 1) / (2.0 * numARows))*PI_d*(threadIdx.x + k*TILE_DIM))*sqrtf(1.0 / (1 + DELTA(1, (threadIdx.x + k*TILE_DIM) + 1)))*sqrtf(2.0 / numARows); }
		
		else											{ As[threadIdx.y][threadIdx.x] = 0.0; }

		if (k*TILE_DIM + threadIdx.y < numARows && Col < numAColumns){ Bs[threadIdx.y][threadIdx.x] = A[(k*TILE_DIM + threadIdx.y)*numAColumns + Col]; }
		else								{ Bs[threadIdx.y][threadIdx.x] = 0.0; }

		__syncthreads();

		for (int n = 0; n < TILE_DIM; ++n) { CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x]; }

		__syncthreads();
	}

	if (Row < numCRows && Col < numCColumns) { C[((blockIdx.y * blockDim.y + threadIdx.y)*numCColumns) + (blockIdx.x*blockDim.x) + threadIdx.x] = CValue; }

}
template <unsigned int TILE_DIM >  __global__ void DCTIII_Column_Kernel_GPUAS(float const * const A, float * const C,
	int numARows, int numAColumns,
	int numCRows, int numCColumns)
{
	float CValue = 0.0f;
	const float  PI_d = 3.141592653589793238462643383279502884f; //pi
	int Row = blockIdx.y*TILE_DIM + threadIdx.y;
	int Col = blockIdx.x*TILE_DIM + threadIdx.x;

	__shared__ float As[TILE_DIM][TILE_DIM];
	__shared__ float Bs[TILE_DIM][TILE_DIM];

	for (int k = 0; k < (TILE_DIM + numARows - 1) / TILE_DIM; k++) {

		if (k*TILE_DIM + threadIdx.x < numARows && Row < numARows)	{ As[threadIdx.y][threadIdx.x] = __cosf(((2 * Row + 1) / (2.0 * numARows))*PI_d*(threadIdx.x + k*TILE_DIM))*sqrtf(1.0 / (1 + DELTA(1, (threadIdx.x + k*TILE_DIM) + 1)))*sqrtf(2.0 / numARows); }
		
		else											{ As[threadIdx.y][threadIdx.x] = 0.0; }

		if (k*TILE_DIM + threadIdx.y < numARows && Col < numAColumns){ Bs[threadIdx.y][threadIdx.x] = A[(k*TILE_DIM + threadIdx.y)*numAColumns + Col]; }
		else								{ Bs[threadIdx.y][threadIdx.x] = 0.0; }

		__syncthreads();

		for (int n = 0; n < TILE_DIM; ++n) { CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x]; }

		__syncthreads();
	}

	if (Row < numCRows && Col < numCColumns) { C[((blockIdx.y * blockDim.y + threadIdx.y)*numCColumns) + (blockIdx.x*blockDim.x) + threadIdx.x] = CValue; }

}

template <unsigned int TILE_DIM >  __global__ void DCTIII_Column_Inverse_Kernel_GPUAS(float const * const A, float * const C,
	int numARows, int numAColumns,
	int numCRows, int numCColumns)
{
	float CValue = 0.0f;
	const float  PI_d = 3.141592653589793238462643383279502884f; //pi
	int Row = blockIdx.y*TILE_DIM + threadIdx.y;
	int Col = blockIdx.x*TILE_DIM + threadIdx.x;

	__shared__ float As[TILE_DIM][TILE_DIM];
	__shared__ float Bs[TILE_DIM][TILE_DIM];

	for (int k = 0; k < (TILE_DIM + numARows - 1) / TILE_DIM; k++) {

		if (k*TILE_DIM + threadIdx.x < numARows && Row < numARows)	{ As[threadIdx.y][threadIdx.x] = __cosf(((2 * (threadIdx.x + k*TILE_DIM) + 1) / (2.0 * numARows))*PI_d*Row)*sqrtf(1.0 / (1 + DELTA(Row + 1, 1) ))*sqrtf(2.0 / numARows); }
		
		else											{ As[threadIdx.y][threadIdx.x] = 0.0; }

		if (k*TILE_DIM + threadIdx.y < numARows && Col < numAColumns){ Bs[threadIdx.y][threadIdx.x] = A[(k*TILE_DIM + threadIdx.y)*numAColumns + Col]; }
		else								{ Bs[threadIdx.y][threadIdx.x] = 0.0; }

		__syncthreads();

		for (int n = 0; n < TILE_DIM; ++n) { CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x]; }

		__syncthreads();
	}

	if (Row < numCRows && Col < numCColumns) { C[((blockIdx.y * blockDim.y + threadIdx.y)*numCColumns) + (blockIdx.x*blockDim.x) + threadIdx.x] = CValue; }

}
template <unsigned int TILE_DIM >  __global__ void DCTIII_Row_Kernel_GPUAS(float const * const A, float * const C,
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

		if (k*TILE_DIM + threadIdx.y < numAColumns && Col < numAColumns)	{ Bs[threadIdx.y][threadIdx.x] = __cosf(((2 * Col + 1) / (2.0 * numAColumns))*PI_d*(threadIdx.y + k*TILE_DIM))*sqrtf(1.0 / (1 + DELTA(1, (threadIdx.y + k*TILE_DIM) + 1)))*sqrtf(2.0 / numAColumns); }
		
		else													{ Bs[threadIdx.y][threadIdx.x] = 0.0; }

		__syncthreads();

		for (int n = 0; n < TILE_DIM; ++n) { CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x]; }

		__syncthreads();
	}

	if (Row < numCRows && Col < numCColumns) { C[((blockIdx.y * blockDim.y + threadIdx.y)*numCColumns) + (blockIdx.x*blockDim.x) + threadIdx.x] = CValue; }

}

template <unsigned int TILE_DIM >  __global__ void DCTIII_Row__InverseKernel_GPUAS(float const * const A, float * const C,
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

		if (k*TILE_DIM + threadIdx.y < numAColumns && Col < numAColumns)	{ Bs[threadIdx.y][threadIdx.x] = __cosf(((2 * (threadIdx.y + k*TILE_DIM) + 1) / (2.0 * numAColumns))*PI_d*Col)*sqrtf(1.0 / (1 + DELTA(Col + 1, 1) ))*sqrtf(2.0 / numAColumns); }
		
		else													{ Bs[threadIdx.y][threadIdx.x] = 0.0; }

		__syncthreads();

		for (int n = 0; n < TILE_DIM; ++n) { CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x]; }

		__syncthreads();
	}

	if (Row < numCRows && Col < numCColumns) { C[((blockIdx.y * blockDim.y + threadIdx.y)*numCColumns) + (blockIdx.x*blockDim.x) + threadIdx.x] = CValue; }

}

template <unsigned int TILE_DIM >  __global__ void DCTIV_Column_Kernel_GPUAS(float const * const A, float * const C,
	int numARows, int numAColumns,
	int numCRows, int numCColumns)
{
	float CValue = 0.0f;
	const float  PI_d = 3.141592653589793238462643383279502884f; //pi
	int Row = blockIdx.y*TILE_DIM + threadIdx.y;
	int Col = blockIdx.x*TILE_DIM + threadIdx.x;

	__shared__ float As[TILE_DIM][TILE_DIM];
	__shared__ float Bs[TILE_DIM][TILE_DIM];

	for (int k = 0; k < (TILE_DIM + numARows - 1) / TILE_DIM; k++) {

		if (k*TILE_DIM + threadIdx.x < numARows && Row < numARows)	{ As[threadIdx.y][threadIdx.x] = __cosf(((2 * (threadIdx.x + k*TILE_DIM) + 1)*PI_d*(2 * Row + 1) / (4.0 * numARows)))*sqrtf(2.0 / numARows); }
			
		else											{ As[threadIdx.y][threadIdx.x] = 0.0; }

		if (k*TILE_DIM + threadIdx.y < numARows && Col < numAColumns){ Bs[threadIdx.y][threadIdx.x] = A[(k*TILE_DIM + threadIdx.y)*numAColumns + Col]; }
		else								{ Bs[threadIdx.y][threadIdx.x] = 0.0; }

		__syncthreads();

		for (int n = 0; n < TILE_DIM; ++n) { CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x]; }

		__syncthreads();
	}

	if (Row < numCRows && Col < numCColumns) { C[((blockIdx.y * blockDim.y + threadIdx.y)*numCColumns) + (blockIdx.x*blockDim.x) + threadIdx.x] = CValue; }

}

template <unsigned int TILE_DIM >  __global__ void DCTIV_Column_Inverse_Kernel_GPUAS(float const * const A, float * const C,
	int numARows, int numAColumns,
	int numCRows, int numCColumns)
{
	float CValue = 0.0f;
	const float  PI_d = 3.141592653589793238462643383279502884f; //pi
	int Row = blockIdx.y*TILE_DIM + threadIdx.y;
	int Col = blockIdx.x*TILE_DIM + threadIdx.x;

	__shared__ float As[TILE_DIM][TILE_DIM];
	__shared__ float Bs[TILE_DIM][TILE_DIM];

	for (int k = 0; k < (TILE_DIM + numARows - 1) / TILE_DIM; k++) {

		if (k*TILE_DIM + threadIdx.x < numARows && Row < numARows)	{ As[threadIdx.y][threadIdx.x] = __cosf(((2 * (threadIdx.x + k*TILE_DIM) + 1)*PI_d*(2 * Row + 1) / (4.0 * numARows)))*sqrtf(2.0 / numARows); }
		
		else											{ As[threadIdx.y][threadIdx.x] = 0.0; }

		if (k*TILE_DIM + threadIdx.y < numARows && Col < numAColumns){ Bs[threadIdx.y][threadIdx.x] = A[(k*TILE_DIM + threadIdx.y)*numAColumns + Col]; }
		else								{ Bs[threadIdx.y][threadIdx.x] = 0.0; }

		__syncthreads();

		for (int n = 0; n < TILE_DIM; ++n) { CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x]; }

		__syncthreads();
	}

	if (Row < numCRows && Col < numCColumns) { C[((blockIdx.y * blockDim.y + threadIdx.y)*numCColumns) + (blockIdx.x*blockDim.x) + threadIdx.x] = CValue; }

}

template <unsigned int TILE_DIM >  __global__ void DCTIV_Row_Kernel_GPUAS(float const * const A, float * const C,
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

		if (k*TILE_DIM + threadIdx.y < numAColumns && Col < numAColumns)	{ Bs[threadIdx.y][threadIdx.x] = __cosf(((2 * (threadIdx.y + k*TILE_DIM) + 1)*PI_d*(2 * Col + 1) / (4.0 * numAColumns)))*sqrtf(2.0 / numAColumns); }
		
		else													{ Bs[threadIdx.y][threadIdx.x] = 0.0; }

		__syncthreads();

		for (int n = 0; n < TILE_DIM; ++n) { CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x]; }

		__syncthreads();
	}

	if (Row < numCRows && Col < numCColumns) { C[((blockIdx.y * blockDim.y + threadIdx.y)*numCColumns) + (blockIdx.x*blockDim.x) + threadIdx.x] = CValue; }

}

template <unsigned int TILE_DIM >  __global__ void DCTIV_Row__InverseKernel_GPUAS(float const * const A, float * const C,
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

		if (k*TILE_DIM + threadIdx.y < numAColumns && Col < numAColumns)	{ Bs[threadIdx.y][threadIdx.x] = __cosf(((2 * (threadIdx.y + k*TILE_DIM) + 1)*PI_d*(2 * Col + 1) / (4.0 * numAColumns)))*sqrtf(2.0 / numAColumns); }
		
		else													{ Bs[threadIdx.y][threadIdx.x] = 0.0; }

		__syncthreads();

		for (int n = 0; n < TILE_DIM; ++n) { CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x]; }

		__syncthreads();
	}

	if (Row < numCRows && Col < numCColumns) { C[((blockIdx.y * blockDim.y + threadIdx.y)*numCColumns) + (blockIdx.x*blockDim.x) + threadIdx.x] = CValue; }

}
// END cosine.................................................................................................................

// sine
template <unsigned int TILE_DIM >  __global__ void DSTI_Column_Kernel_GPUAS(float const * const A, float * const C,
	int numARows, int numAColumns,
	int numCRows, int numCColumns)
{
	float CValue = 0.0f;
	const float  PI_d = 3.141592653589793238462643383279502884f; //pi
	int Row = blockIdx.y*TILE_DIM + threadIdx.y;
	int Col = blockIdx.x*TILE_DIM + threadIdx.x;

	__shared__ float As[TILE_DIM][TILE_DIM];
	__shared__ float Bs[TILE_DIM][TILE_DIM];

	for (int k = 0; k < (TILE_DIM + numARows - 1) / TILE_DIM; k++) {
                                                                     //As[threadIdx.y][threadIdx.x] = __cosf((Row*PI_d*(threadIdx.x + (k*TILE_DIM)) / (numARows - 1)))*sqrtf(1.0 / (1 + DELTA((threadIdx.x + (k*TILE_DIM)) + 1, 1) + DELTA((threadIdx.x + (k*TILE_DIM)) + 1, numARows)))*sqrtf(1.0 / (1 + DELTA(1, Row + 1) + DELTA(numARows, Row + 1)))*sqrtf(2.0 / numARows)
		if (k*TILE_DIM + threadIdx.x < numARows && Row < numARows)	{ As[threadIdx.y][threadIdx.x] = __sinf(((((threadIdx.x + k*TILE_DIM)+1)*PI_d*(Row+1)) / (numARows + 1)))*sqrtf(2.0 / (numARows+1)); }
		//As[threadIdx.y][threadIdx.x] = A[Row*ACols + k*TILE_DIM + threadIdx.x];
		else											{ As[threadIdx.y][threadIdx.x] = 0.0; }

		if (k*TILE_DIM + threadIdx.y < numARows && Col < numAColumns){ Bs[threadIdx.y][threadIdx.x] = A[(k*TILE_DIM + threadIdx.y)*numAColumns + Col]; }
		else								{ Bs[threadIdx.y][threadIdx.x] = 0.0; }

		__syncthreads();

		for (int n = 0; n < TILE_DIM; ++n) { CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x]; }

		__syncthreads();
	}

	if (Row < numCRows && Col < numCColumns) { C[((blockIdx.y * blockDim.y + threadIdx.y)*numCColumns) + (blockIdx.x*blockDim.x) + threadIdx.x] = CValue; }

}

template <unsigned int TILE_DIM >  __global__ void DSTI_Column_Inverse_Kernel_GPUAS(float const * const A, float * const C,
	int numARows, int numAColumns,
	int numCRows, int numCColumns)
{
	float CValue = 0.0f;
	const float  PI_d = 3.141592653589793238462643383279502884f; //pi
	int Row = blockIdx.y*TILE_DIM + threadIdx.y;
	int Col = blockIdx.x*TILE_DIM + threadIdx.x;

	__shared__ float As[TILE_DIM][TILE_DIM];
	__shared__ float Bs[TILE_DIM][TILE_DIM];

	for (int k = 0; k < (TILE_DIM + numARows - 1) / TILE_DIM; k++) {
                                                                     //As[threadIdx.y][threadIdx.x] = __cosf((Row*PI_d*(threadIdx.x + (k*TILE_DIM)) / (numARows - 1)))*sqrtf(1.0 / (1 + DELTA((threadIdx.x + (k*TILE_DIM)) + 1, 1) + DELTA((threadIdx.x + (k*TILE_DIM)) + 1, numARows)))*sqrtf(1.0 / (1 + DELTA(1, Row + 1) + DELTA(numARows, Row + 1)))*sqrtf(2.0 / numARows);
		if (k*TILE_DIM + threadIdx.x < numARows && Row < numARows)	{ As[threadIdx.y][threadIdx.x] = __sinf(((((threadIdx.x + k*TILE_DIM) + 1)*PI_d*(Row + 1)) / (numARows + 1)))*sqrtf(2.0 / (numARows + 1)); }
		//As[threadIdx.y][threadIdx.x] = A[Row*ACols + k*TILE_DIM + threadIdx.x];
		else											{ As[threadIdx.y][threadIdx.x] = 0.0; }

		if (k*TILE_DIM + threadIdx.y < numARows && Col < numAColumns){ Bs[threadIdx.y][threadIdx.x] = A[(k*TILE_DIM + threadIdx.y)*numAColumns + Col]; }
		else								{ Bs[threadIdx.y][threadIdx.x] = 0.0; }

		__syncthreads();

		for (int n = 0; n < TILE_DIM; ++n) { CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x]; }

		__syncthreads();
	}

	if (Row < numCRows && Col < numCColumns) { C[((blockIdx.y * blockDim.y + threadIdx.y)*numCColumns) + (blockIdx.x*blockDim.x) + threadIdx.x] = CValue; }

}
template <unsigned int TILE_DIM >  __global__ void DSTI_Row_Kernel_GPUAS(float const * const A, float * const C,
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
                                                                             //Bs[threadIdx.y][threadIdx.x] = __cosf(((threadIdx.y + k*TILE_DIM)*PI_d*Col / (numAColumns - 1)))*sqrtf(1.0 / (1 + DELTA(Col + 1, 1) + DELTA(Col + 1, numAColumns)))*sqrtf(1.0 / (1 + DELTA(1, (threadIdx.y + k*TILE_DIM) + 1) + DELTA(numAColumns, (threadIdx.y + k*TILE_DIM) + 1)))*sqrtf(2.0 / numAColumns);
		if (k*TILE_DIM + threadIdx.y < numAColumns && Col < numAColumns)	{ Bs[threadIdx.y][threadIdx.x] = __sinf(((((threadIdx.y + k*TILE_DIM)+1)*PI_d*(Col+1)) / (numAColumns + 1)))*sqrtf(2.0 / (numAColumns+1)); }
		//Bs[threadIdx.y][threadIdx.x] = B[(k*TILE_DIM + threadIdx.y)*BCols + Col];
		else													{ Bs[threadIdx.y][threadIdx.x] = 0.0; }

		__syncthreads();

		for (int n = 0; n < TILE_DIM; ++n) { CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x]; }

		__syncthreads();
	}

	if (Row < numCRows && Col < numCColumns) { C[((blockIdx.y * blockDim.y + threadIdx.y)*numCColumns) + (blockIdx.x*blockDim.x) + threadIdx.x] = CValue; }

}

template <unsigned int TILE_DIM >  __global__ void DSTI_Row__InverseKernel_GPUAS(float const * const A, float * const C,
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
                                                                              //Bs[threadIdx.y][threadIdx.x] = __cosf(((threadIdx.y + k*TILE_DIM)*PI_d*Col / (numAColumns - 1)))*sqrtf(1.0 / (1 + DELTA(Col + 1, 1) + DELTA(Col + 1, numAColumns)))*sqrtf(1.0 / (1 + DELTA(1, (threadIdx.y + k*TILE_DIM) + 1) + DELTA(numAColumns, (threadIdx.y + k*TILE_DIM) + 1)))*sqrtf(2.0 / numAColumns);
		if (k*TILE_DIM + threadIdx.y < numAColumns && Col < numAColumns)	{ Bs[threadIdx.y][threadIdx.x] = __sinf(((((threadIdx.y + k*TILE_DIM) + 1)*PI_d*(Col + 1)) / (numAColumns + 1)))*sqrtf(2.0 / (numAColumns + 1)); }
		//Bs[threadIdx.y][threadIdx.x] = B[(k*TILE_DIM + threadIdx.y)*BCols + Col];
		else													{ Bs[threadIdx.y][threadIdx.x] = 0.0; }

		__syncthreads();

		for (int n = 0; n < TILE_DIM; ++n) { CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x]; }

		__syncthreads();
	}

	if (Row < numCRows && Col < numCColumns) { C[((blockIdx.y * blockDim.y + threadIdx.y)*numCColumns) + (blockIdx.x*blockDim.x) + threadIdx.x] = CValue; }

}
template <unsigned int TILE_DIM >  __global__ void DSTII_Row_Kernel_GPUAS(float const * const A, float * const C,
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
                                                                             //Bs[threadIdx.y][threadIdx.x] = __cosf(((2 * (threadIdx.y + k*TILE_DIM) + 1) / (2.0 * numAColumns))*PI_d*Col)*sqrtf(1.0 / (1 + DELTA(Col + 1, 1)))*sqrtf(2.0 / numAColumns);
		if (k*TILE_DIM + threadIdx.y < numAColumns && Col < numAColumns)	{ Bs[threadIdx.y][threadIdx.x] = __sinf((((threadIdx.y + k*TILE_DIM) + 0.5)*PI_d*(Col + 1)) / (numAColumns))*sqrtf((2.0 - DELTA(Col + 1, numAColumns)) / (numAColumns));  }
		//Bs[threadIdx.y][threadIdx.x] = B[(k*TILE_DIM + threadIdx.y)*BCols + Col];
		else													{ Bs[threadIdx.y][threadIdx.x] = 0.0; }

		__syncthreads();

		for (int n = 0; n < TILE_DIM; ++n) { CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x]; }

		__syncthreads();
	}

	if (Row < numCRows && Col < numCColumns) { C[((blockIdx.y * blockDim.y + threadIdx.y)*numCColumns) + (blockIdx.x*blockDim.x) + threadIdx.x] = CValue; }

}
template <unsigned int TILE_DIM >  __global__ void DSTII_Row__InverseKernel_GPUAS(float const * const A, float * const C,
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
                                                                             //Bs[threadIdx.y][threadIdx.x] = __cosf(((2 * Col + 1) / (2.0 * numAColumns))*PI_d*(threadIdx.y + k*TILE_DIM))*sqrtf(1.0 / (1 + DELTA(1, (threadIdx.y + k*TILE_DIM) + 1)))*sqrtf(2.0 / numAColumns);
		if (k*TILE_DIM + threadIdx.y < numAColumns && Col < numAColumns)	{ Bs[threadIdx.y][threadIdx.x] = __sinf((((threadIdx.y + k*TILE_DIM) + 1)*PI_d*(Col + 0.5)) / (numAColumns))*sqrtf(2.0 / (numAColumns))*sqrtf(1.0 / (1 + DELTA(numAColumns, (threadIdx.y + k*TILE_DIM) + 1))); }
		//Bs[threadIdx.y][threadIdx.x] = B[(k*TILE_DIM + threadIdx.y)*BCols + Col];
		else													{ Bs[threadIdx.y][threadIdx.x] = 0.0; }

		__syncthreads();

		for (int n = 0; n < TILE_DIM; ++n) { CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x]; }

		__syncthreads();
	}

	if (Row < numCRows && Col < numCColumns) { C[((blockIdx.y * blockDim.y + threadIdx.y)*numCColumns) + (blockIdx.x*blockDim.x) + threadIdx.x] = CValue; }

}
template <unsigned int TILE_DIM >  __global__ void DSTII_Column_Kernel_GPUAS(float const * const A, float * const C,
	int numARows, int numAColumns,
	int numCRows, int numCColumns)
{
	float CValue = 0.0f;
	const float  PI_d = 3.141592653589793238462643383279502884f; //pi
	int Row = blockIdx.y*TILE_DIM + threadIdx.y;
	int Col = blockIdx.x*TILE_DIM + threadIdx.x;

	__shared__ float As[TILE_DIM][TILE_DIM];
	__shared__ float Bs[TILE_DIM][TILE_DIM];

	for (int k = 0; k < (TILE_DIM + numARows - 1) / TILE_DIM; k++) {
                                                              //As[threadIdx.y][threadIdx.x] = __cosf(((2 * (threadIdx.x + (k*TILE_DIM)) + 1) / (2.0 * numARows))*PI_d*Row)*sqrtf(1.0 / (1 + DELTA(1, Row + 1)))*sqrtf(2.0 / numARows);      
		if (k*TILE_DIM + threadIdx.x < numARows && Row < numARows)	{ As[threadIdx.y][threadIdx.x] = __sinf((((threadIdx.x + k*TILE_DIM) + 0.5)*PI_d*(Row + 1)) / (numARows))*sqrtf((2.0 - DELTA(Row + 1, numARows)) / (numARows)); }
		
		else											{ As[threadIdx.y][threadIdx.x] = 0.0; }

		if (k*TILE_DIM + threadIdx.y < numARows && Col < numAColumns){ Bs[threadIdx.y][threadIdx.x] = A[(k*TILE_DIM + threadIdx.y)*numAColumns + Col]; }
		else								{ Bs[threadIdx.y][threadIdx.x] = 0.0; }

		__syncthreads();

		for (int n = 0; n < TILE_DIM; ++n) { CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x]; }

		__syncthreads();
	}

	if (Row < numCRows && Col < numCColumns) { C[((blockIdx.y * blockDim.y + threadIdx.y)*numCColumns) + (blockIdx.x*blockDim.x) + threadIdx.x] = CValue; }

}
template <unsigned int TILE_DIM >  __global__ void DSTII_Column_Inverse_Kernel_GPUAS(float const * const A, float * const C,
	int numARows, int numAColumns,
	int numCRows, int numCColumns)
{
	float CValue = 0.0f;
	const float  PI_d = 3.141592653589793238462643383279502884f; //pi
	int Row = blockIdx.y*TILE_DIM + threadIdx.y;
	int Col = blockIdx.x*TILE_DIM + threadIdx.x;

	__shared__ float As[TILE_DIM][TILE_DIM];
	__shared__ float Bs[TILE_DIM][TILE_DIM];

	for (int k = 0; k < (TILE_DIM + numARows - 1) / TILE_DIM; k++) {
                                                                     //As[threadIdx.y][threadIdx.x] = __cosf(((2 * (threadIdx.x + (k*TILE_DIM)) + 1) / (2.0 * numARows))*PI_d*Row)*sqrtf(1.0 / (1 + DELTA(1, Row + 1)))*sqrtf(2.0 / numARows);
		if (k*TILE_DIM + threadIdx.x < numARows && Row < numARows)	{ As[threadIdx.y][threadIdx.x] = __sinf((((threadIdx.x + k*TILE_DIM) + 1)*PI_d*(Row + 0.5)) / (numARows))*sqrtf(2.0 / (numARows))*sqrtf(1.0 / (1 + DELTA(numARows, (threadIdx.x + k*TILE_DIM) + 1))); }
		
		else											{ As[threadIdx.y][threadIdx.x] = 0.0; }

		if (k*TILE_DIM + threadIdx.y < numARows && Col < numAColumns){ Bs[threadIdx.y][threadIdx.x] = A[(k*TILE_DIM + threadIdx.y)*numAColumns + Col]; }
		else								{ Bs[threadIdx.y][threadIdx.x] = 0.0; }

		__syncthreads();

		for (int n = 0; n < TILE_DIM; ++n) { CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x]; }

		__syncthreads();
	}

	if (Row < numCRows && Col < numCColumns) { C[((blockIdx.y * blockDim.y + threadIdx.y)*numCColumns) + (blockIdx.x*blockDim.x) + threadIdx.x] = CValue; }

}
template <unsigned int TILE_DIM >  __global__ void DSTIII_Column_Kernel_GPUAS(float const * const A, float * const C,
	int numARows, int numAColumns,
	int numCRows, int numCColumns)
{
	float CValue = 0.0f;
	const float  PI_d = 3.141592653589793238462643383279502884f; //pi
	int Row = blockIdx.y*TILE_DIM + threadIdx.y;
	int Col = blockIdx.x*TILE_DIM + threadIdx.x;

	__shared__ float As[TILE_DIM][TILE_DIM];
	__shared__ float Bs[TILE_DIM][TILE_DIM];

	for (int k = 0; k < (TILE_DIM + numARows - 1) / TILE_DIM; k++) {
                                                                    //As[threadIdx.y][threadIdx.x] = __cosf(((2 * (threadIdx.x + (k*TILE_DIM)) + 1) / (2.0 * numARows))*PI_d*Row)*sqrtf(1.0 / (1 + DELTA(1, Row + 1)))*sqrtf(2.0 / numARows);
		if (k*TILE_DIM + threadIdx.x < numARows && Row < numARows)	{ As[threadIdx.y][threadIdx.x] = __sinf((((threadIdx.x + k*TILE_DIM) + 1)*PI_d*(Row + 0.5)) / (numARows))*sqrtf(2.0 / (numARows))*sqrtf(1.0 / (1 + DELTA(numARows, (threadIdx.x + k*TILE_DIM) + 1))); }
		//As[threadIdx.y][threadIdx.x] = A[Row*ACols + k*TILE_DIM + threadIdx.x];
		else											{ As[threadIdx.y][threadIdx.x] = 0.0; }

		if (k*TILE_DIM + threadIdx.y < numARows && Col < numAColumns){ Bs[threadIdx.y][threadIdx.x] = A[(k*TILE_DIM + threadIdx.y)*numAColumns + Col]; }
		else								{ Bs[threadIdx.y][threadIdx.x] = 0.0; }

		__syncthreads();

		for (int n = 0; n < TILE_DIM; ++n) { CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x]; }

		__syncthreads();
	}

	if (Row < numCRows && Col < numCColumns) { C[((blockIdx.y * blockDim.y + threadIdx.y)*numCColumns) + (blockIdx.x*blockDim.x) + threadIdx.x] = CValue; }

}

template <unsigned int TILE_DIM >  __global__ void DSTIII_Column_Inverse_Kernel_GPUAS(float const * const A, float * const C,
	int numARows, int numAColumns,
	int numCRows, int numCColumns)
{
	float CValue = 0.0f;
	const float  PI_d = 3.141592653589793238462643383279502884f; //pi
	int Row = blockIdx.y*TILE_DIM + threadIdx.y;
	int Col = blockIdx.x*TILE_DIM + threadIdx.x;

	__shared__ float As[TILE_DIM][TILE_DIM];
	__shared__ float Bs[TILE_DIM][TILE_DIM];

	for (int k = 0; k < (TILE_DIM + numARows - 1) / TILE_DIM; k++) {
                                                                    //As[threadIdx.y][threadIdx.x] = __cosf(((2 * Row + 1) / (2.0 * numARows))*PI_d*(threadIdx.x + (k*TILE_DIM)))*sqrtf(1.0 / (1 + DELTA((threadIdx.x + (k*TILE_DIM)) + 1, 1)))*sqrtf(2.0 / numARows);
		if (k*TILE_DIM + threadIdx.x < numARows && Row < numARows)	{ As[threadIdx.y][threadIdx.x] = __sinf((((threadIdx.x + k*TILE_DIM) + 0.5)*PI_d*(Row + 1)) / (numARows))*sqrtf((2.0 - DELTA(Row + 1, numARows)) / (numARows)); }
		
		else											{ As[threadIdx.y][threadIdx.x] = 0.0; }

		if (k*TILE_DIM + threadIdx.y < numARows && Col < numAColumns){ Bs[threadIdx.y][threadIdx.x] = A[(k*TILE_DIM + threadIdx.y)*numAColumns + Col]; }
		else								{ Bs[threadIdx.y][threadIdx.x] = 0.0; }

		__syncthreads();

		for (int n = 0; n < TILE_DIM; ++n) { CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x]; }

		__syncthreads();
	}

	if (Row < numCRows && Col < numCColumns) { C[((blockIdx.y * blockDim.y + threadIdx.y)*numCColumns) + (blockIdx.x*blockDim.x) + threadIdx.x] = CValue; }

}
template <unsigned int TILE_DIM >  __global__ void DSTIII_Row_Kernel_GPUAS(float const * const A, float * const C,
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
                                                                             //Bs[threadIdx.y][threadIdx.x] = __cosf(((2 * Col + 1) / (2.0 * numAColumns))*PI_d*(threadIdx.y + k*TILE_DIM))*sqrtf(1.0 / (1 + DELTA(1, (threadIdx.y + k*TILE_DIM) + 1)))*sqrtf(2.0 / numAColumns);
		if (k*TILE_DIM + threadIdx.y < numAColumns && Col < numAColumns)	{ Bs[threadIdx.y][threadIdx.x] = __sinf((((threadIdx.y + k*TILE_DIM) + 1)*PI_d*(Col + 0.5)) / (numAColumns))*sqrtf(2.0 / (numAColumns))*sqrtf(1.0 / (1 + DELTA(numAColumns, (threadIdx.y + k*TILE_DIM) + 1))); }
		
		else													{ Bs[threadIdx.y][threadIdx.x] = 0.0; }

		__syncthreads();

		for (int n = 0; n < TILE_DIM; ++n) { CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x]; }

		__syncthreads();
	}

	if (Row < numCRows && Col < numCColumns) { C[((blockIdx.y * blockDim.y + threadIdx.y)*numCColumns) + (blockIdx.x*blockDim.x) + threadIdx.x] = CValue; }

}

template <unsigned int TILE_DIM >  __global__ void DSTIII_Row__InverseKernel_GPUAS(float const * const A, float * const C,
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
                                                                           //Bs[threadIdx.y][threadIdx.x] = __cosf(((2 * (threadIdx.y + k*TILE_DIM) + 1) / (2.0 * numAColumns))*PI_d*Col)*sqrtf(1.0 / (1 + DELTA(Col + 1, 1) ))*sqrtf(2.0 / numAColumns);
		if (k*TILE_DIM + threadIdx.y < numAColumns && Col < numAColumns)	{ Bs[threadIdx.y][threadIdx.x] = __sinf((((threadIdx.y + k*TILE_DIM) + 0.5)*PI_d*(Col + 1)) / (numAColumns))*sqrtf((2.0 - DELTA(Col + 1, numAColumns)) / (numAColumns)); }
	
		else													{ Bs[threadIdx.y][threadIdx.x] = 0.0; }

		__syncthreads();

		for (int n = 0; n < TILE_DIM; ++n) { CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x]; }

		__syncthreads();
	}

	if (Row < numCRows && Col < numCColumns) { C[((blockIdx.y * blockDim.y + threadIdx.y)*numCColumns) + (blockIdx.x*blockDim.x) + threadIdx.x] = CValue; }

}

template <unsigned int TILE_DIM >  __global__ void DSTIV_Column_Kernel_GPUAS(float const * const A, float * const C,
	int numARows, int numAColumns,
	int numCRows, int numCColumns)
{
	float CValue = 0.0f;
	const float  PI_d = 3.141592653589793238462643383279502884f; //pi
	int Row = blockIdx.y*TILE_DIM + threadIdx.y;
	int Col = blockIdx.x*TILE_DIM + threadIdx.x;

	__shared__ float As[TILE_DIM][TILE_DIM];
	__shared__ float Bs[TILE_DIM][TILE_DIM];

	for (int k = 0; k < (TILE_DIM + numARows - 1) / TILE_DIM; k++) {
                                                                    //As[threadIdx.y][threadIdx.x] = __cosf(((2 * Row + 1)*PI_d*(2 * (threadIdx.x + (k*TILE_DIM)) + 1) / (4.0 * numARows)))*sqrtf(2.0 / numARows);
		if (k*TILE_DIM + threadIdx.x < numARows && Row < numARows)	{ As[threadIdx.y][threadIdx.x] = __sinf((((threadIdx.x + k*TILE_DIM) + 0.5)*PI_d*(Row + 0.5)) / (numARows))*sqrtf(2.0 / (numARows)); }
	
		else											{ As[threadIdx.y][threadIdx.x] = 0.0; }

		if (k*TILE_DIM + threadIdx.y < numARows && Col < numAColumns){ Bs[threadIdx.y][threadIdx.x] = A[(k*TILE_DIM + threadIdx.y)*numAColumns + Col]; }
		else								{ Bs[threadIdx.y][threadIdx.x] = 0.0; }

		__syncthreads();

		for (int n = 0; n < TILE_DIM; ++n) { CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x]; }

		__syncthreads();
	}

	if (Row < numCRows && Col < numCColumns) { C[((blockIdx.y * blockDim.y + threadIdx.y)*numCColumns) + (blockIdx.x*blockDim.x) + threadIdx.x] = CValue; }

}

template <unsigned int TILE_DIM >  __global__ void DSTIV_Column_Inverse_Kernel_GPUAS(float const * const A, float * const C,
	int numARows, int numAColumns,
	int numCRows, int numCColumns)
{
	float CValue = 0.0f;
	const float  PI_d = 3.141592653589793238462643383279502884f; //pi
	int Row = blockIdx.y*TILE_DIM + threadIdx.y;
	int Col = blockIdx.x*TILE_DIM + threadIdx.x;

	__shared__ float As[TILE_DIM][TILE_DIM];
	__shared__ float Bs[TILE_DIM][TILE_DIM];

	for (int k = 0; k < (TILE_DIM + numARows - 1) / TILE_DIM; k++) {
                                                                     //As[threadIdx.y][threadIdx.x] = __cosf(((2 * Row + 1)*PI_d*(2 * (threadIdx.x + (k*TILE_DIM)) + 1) / (4.0 * numARows)))*sqrtf(2.0 / numARows);
		if (k*TILE_DIM + threadIdx.x < numARows && Row < numARows)	{ As[threadIdx.y][threadIdx.x] = __sinf((((threadIdx.x + k*TILE_DIM) + 0.5)*PI_d*(Row + 0.5)) / (numARows))*sqrtf(2.0 / (numARows)); }
		
		else											{ As[threadIdx.y][threadIdx.x] = 0.0; }

		if (k*TILE_DIM + threadIdx.y < numARows && Col < numAColumns){ Bs[threadIdx.y][threadIdx.x] = A[(k*TILE_DIM + threadIdx.y)*numAColumns + Col]; }
		else								{ Bs[threadIdx.y][threadIdx.x] = 0.0; }

		__syncthreads();

		for (int n = 0; n < TILE_DIM; ++n) { CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x]; }

		__syncthreads();
	}

	if (Row < numCRows && Col < numCColumns) { C[((blockIdx.y * blockDim.y + threadIdx.y)*numCColumns) + (blockIdx.x*blockDim.x) + threadIdx.x] = CValue; }

}

template <unsigned int TILE_DIM >  __global__ void DSTIV_Row_Kernel_GPUAS(float const * const A, float * const C,
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

template <unsigned int TILE_DIM >  __global__ void DSTIV_Row__InverseKernel_GPUAS(float const * const A, float * const C,
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
// END sine

///////////////////////////////////////////////////////////sine
#endif // #ifndef _DISCRETE_TRANSFORMS_KERNEL_H_
