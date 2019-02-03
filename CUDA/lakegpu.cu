/*
Group info:
nphabia Niklesh Phabiani
rtnaik	Rohit Naik
anjain2 Akshay Narendra Jain
*/

#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>

#define __DEBUG

#define CUDA_CALL( err )     __cudaSafeCall( err, __FILE__, __LINE__ )
#define CUDA_CHK_ERR() __cudaCheckError(__FILE__,__LINE__)

#define TSCALE 1.0
#define VSQR 0.1

/**************************************
* void __cudaSafeCall(cudaError err, const char *file, const int line)
* void __cudaCheckError(const char *file, const int line)
*
* These routines were taken from the GPU Computing SDK
* (http://developer.nvidia.com/gpu-computing-sdk) include file "cutil.h"
**************************************/
inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef __DEBUG

#pragma warning( push )
#pragma warning( disable: 4127 ) // Prevent warning on do-while(0);
  do
  {
    if ( cudaSuccess != err )
    {
      fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
              file, line, cudaGetErrorString( err ) );
      exit( -1 );
    }
  } while ( 0 );
#pragma warning( pop )
#endif  // __DEBUG
  return;
}

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef __DEBUG
#pragma warning( push )
#pragma warning( disable: 4127 ) // Prevent warning on do-while(0);
  do
  {
    cudaError_t err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
      fprintf( stderr, "cudaCheckError() failed at %s:%i : %s.\n",
               file, line, cudaGetErrorString( err ) );
      exit( -1 );
    }
    // More careful checking. However, this will affect performance.
    // Comment if not needed.
    /*err = cudaThreadSynchronize();
    if( cudaSuccess != err )
    {
      fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s.\n",
               file, line, cudaGetErrorString( err ) );
      exit( -1 );
    }*/
  } while ( 0 );
#pragma warning( pop )
#endif // __DEBUG
  return;
}

/* utility functions for lake calculations */
__device__ double f_gpu(double p, double t)
{
  return -expf(-TSCALE * t) * p;
}

// Evolve kernel function to run on GPU
__global__ void evolve_gpu(double *un, double *uc, double *uo, double *pebbles, int n, double h, double dt, double t) {
  int i, j, idx;
  i = blockIdx.x * blockDim.x + threadIdx.x;
  j = blockIdx.y * blockDim.y + threadIdx.y;

  idx = i * n + j;

  if(i == 0 || i == n - 1 || j == 0 || j == n - 1)
    un[idx] = 0.;
  else
  {
    // 5-point stencil formula
    // idx-1 -> WEST; idx+1 -> EAST; idx-n -> NORTH; idx+n -> SOUTH
    /*un[idx] = 2*uc[idx] - uo[idx] + VSQR *(dt * dt) *((uc[idx-1] + uc[idx+1] +
                uc[idx + n] + uc[idx - n] - 4 * uc[idx])/(h * h) + f_gpu(pebbles[idx],t));*/

    // 9-point stencil formula
    // idx-n-1 -> NORTHWEST; idx-n+1 -> NORTHEAST; idx+n-1 -> SOUTHWEST; idx+n+1 -> SOUTHEAST
    un[idx] = 2*uc[idx] - uo[idx] + VSQR *(dt * dt) *(( uc[idx-1] + uc[idx+1] +
                uc[idx-n] + uc[idx+n] + 0.25*(uc[idx-n-1] + uc[idx-n+1] + uc[idx+n-1] + uc[idx+n+1])-
                5 * uc[idx])/(h * h) + f_gpu(pebbles[idx],t));
  }
}

int tpdt_gpu(double *t, double dt, double tf)
{
  if((*t) + dt > tf) return 0;
  (*t) = (*t) + dt;
  return 1;
}

void run_gpu(double *u, double *u0, double *u1, double *pebbles, int n, double h, double end_time, int nthreads)
{
	cudaEvent_t kstart, kstop;
	float ktime;

	/* HW2: Define your local variables here */
  // Calculate the number of blocks and threads.
  double t = 0.;
  double dt = h / 2.;
  int narea = n * n;

  // n: npoints (grid size)
  dim3 no_blocks(n/nthreads, n/nthreads);
  dim3 threads_per_block(nthreads, nthreads);

        /* Set up device timers */
	CUDA_CALL(cudaSetDevice(0));
	CUDA_CALL(cudaEventCreate(&kstart));
	CUDA_CALL(cudaEventCreate(&kstop));

	/* HW2: Add CUDA kernel call preperation code here */
  // Device copies
  double *d_uo;
  double *d_uc;
  double *d_un;
  double *d_pebb;

  // Allocating memory for device copies
  cudaMalloc((double **)&d_uo, narea * sizeof(double));
  cudaMalloc((double **)&d_uc, narea * sizeof(double));
  cudaMalloc((double **)&d_un, narea * sizeof(double));
  cudaMalloc((double **)&d_pebb, narea * sizeof(double));

	/* Start GPU computation timer */
	CUDA_CALL(cudaEventRecord(kstart, 0));

  // Copying host values to device so that the device can then operate using these values
  cudaMemcpy(d_uo, u0, sizeof(double) * narea, cudaMemcpyHostToDevice);
  cudaMemcpy(d_uc, u1, sizeof(double) * narea, cudaMemcpyHostToDevice);
  cudaMemcpy(d_pebb, pebbles, sizeof(double) * narea, cudaMemcpyHostToDevice);

  /* HW2: Add main lake simulation loop here */
  while(1)
  {
    // GPU kernel function call to start computations on GPU
    evolve_gpu<<<no_blocks, threads_per_block>>>(d_un, d_uc, d_uo, d_pebb, n, h, dt, t);

    // Copying evolved values so that they can be used for the next iteration of evolve
    cudaMemcpy(d_uo, d_uc, sizeof(double) * narea, cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_uc, d_un, sizeof(double) * narea, cudaMemcpyDeviceToDevice);

    if(!tpdt_gpu(&t,dt,end_time)) break;
  }

  /* Copy back results to host */
  // from d_u (2d) to u (1d)
  // Copying latest evolve value from device to host after end time is reached
  cudaMemcpy(u, d_un, sizeof(double) * narea, cudaMemcpyDeviceToHost);

        /* Stop GPU computation timer */
	CUDA_CALL(cudaEventRecord(kstop, 0));
	CUDA_CALL(cudaEventSynchronize(kstop));
	CUDA_CALL(cudaEventElapsedTime(&ktime, kstart, kstop));
	printf("GPU computation: %f msec\n", ktime);

	/* HW2: Add post CUDA kernel call processing and cleanup here */
  cudaFree(d_uo);
  cudaFree(d_uc);
  cudaFree(d_un);
  cudaFree(d_pebb);

	/* timer cleanup */
	CUDA_CALL(cudaEventDestroy(kstart));
	CUDA_CALL(cudaEventDestroy(kstop));
}
