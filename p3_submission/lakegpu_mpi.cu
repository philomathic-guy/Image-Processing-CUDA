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
#include <mpi.h>

#define __DEBUG

#define CUDA_CALL( err )     __cudaSafeCall( err, __FILE__, __LINE__ )
#define CUDA_CHK_ERR() __cudaCheckError(__FILE__,__LINE__)

#define STENCIL 9
#define TSCALE 1.0
#define VSQR 0.1

//__global__ void evolve9pt_gpu(double **un, double **uc, double **uo, double **pebbles, int n, double h, double dt, double t);

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

//
__global__ void evolve_gpu(double *un, double *uc, double *uo, double *pebbles, int node_thread_dim, double h, double dt, double t, int rank,
  double *d_uc_received_h, double *d_uc_received_v, double d_uc_received_d) {
  int i, j, idx;
  i = blockIdx.x * blockDim.x + threadIdx.x;
  j = blockIdx.y * blockDim.y + threadIdx.y;

  idx = i * node_thread_dim + j;

  if(rank == 0) {
    // If they are boundary points assign 0 to them.
    if(i == 0 || j == 0) {
      un[idx] = 0.;
    }
    else {
      // If it is the point which needs values from neighbors and diagonals.
      if (i == node_thread_dim-1 && j == node_thread_dim-1) {
        // Get from diagonal and horizontal and vertical.
        un[idx] = 2*uc[idx] - uo[idx] + VSQR *(dt * dt) *(( uc[idx-1] +
                  d_uc_received_v[i] + uc[idx-node_thread_dim] + d_uc_received_h[i] +
                  0.25*(uc[idx-node_thread_dim-1] + d_uc_received_v[i-1] +
                  d_uc_received_h[i-1] + d_uc_received_d)- 5 *
                  uc[idx])/(h * h) + f_gpu(pebbles[idx],t));
      }
      // If it is the point which takes values from horizontal received array.
      else if (i == node_thread_dim-1 && j != node_thread_dim-1) {
        // Use horizontal.
        un[idx] = 2*uc[idx] - uo[idx] + VSQR *(dt * dt) *(( uc[idx-1] +
                  uc[idx+1] + uc[idx-node_thread_dim] + d_uc_received_h[j] +
                  0.25*(uc[idx-node_thread_dim-1] + uc[idx-node_thread_dim+1] + d_uc_received_h[j-1] +
                  d_uc_received_h[j+1]) - 5 * uc[idx])/(h * h) + f_gpu(pebbles[idx],t));
      }
      // If it is the point which takes values from the vertical received array.
      else if (j == node_thread_dim-1 && i != node_thread_dim-1) {
        // Use vertical.
        un[idx] = 2*uc[idx] - uo[idx] + VSQR *(dt * dt) *(( uc[idx-1] +
                  d_uc_received_v[i] + uc[idx-node_thread_dim] + uc[idx+node_thread_dim] +
                  0.25*(uc[idx-node_thread_dim-1] + d_uc_received_v[i-1] + uc[idx+node_thread_dim-1] +
                    d_uc_received_v[i+1]) - 5 * uc[idx])/(h * h) + f_gpu(pebbles[idx],t));
      }
      // If it is any other normal point.
      else {
        un[idx] = 2*uc[idx] - uo[idx] + VSQR *(dt * dt) *(( uc[idx-1] +
          uc[idx+1] + uc[idx-node_thread_dim] + uc[idx+node_thread_dim] +
          0.25*(uc[idx-node_thread_dim-1] + uc[idx-node_thread_dim+1] +
            uc[idx+node_thread_dim-1] + uc[idx+node_thread_dim+1])- 5 *
            uc[idx])/(h * h) + f_gpu(pebbles[idx],t));
      }
    }
  } else if(rank == 1) {
    if(i == 0 || j == node_thread_dim-1) {
      un[idx] = 0.;
    }
    else {
      if (i == node_thread_dim-1 && j == 0) {
        // Get from diagonal and horizontal and vertical.
        un[idx] = 2*uc[idx] - uo[idx] + VSQR *(dt * dt) *(( d_uc_received_v[i] +
                  uc[idx+1] + uc[idx-node_thread_dim] + d_uc_received_h[j] +
                  0.25*(d_uc_received_v[i-1] + uc[idx-node_thread_dim+1] +
                  d_uc_received_d + d_uc_received_h[j+1])- 5 *
                  uc[idx])/(h * h) + f_gpu(pebbles[idx],t));
      } else if (i == node_thread_dim-1 && j != 0) {
        // Use horizontal.
        un[idx] = 2*uc[idx] - uo[idx] + VSQR *(dt * dt) *(( uc[idx-1] +
                  uc[idx+1] + uc[idx-node_thread_dim] + d_uc_received_h[j] +
                  0.25*(uc[idx-node_thread_dim-1] + uc[idx-node_thread_dim+1] + d_uc_received_h[j-1] +
                  d_uc_received_h[j+1]) - 5 * uc[idx])/(h * h) + f_gpu(pebbles[idx],t));
      } else if (j == 0 && i != node_thread_dim-1) {
        // Use vertical.
        un[idx] = 2*uc[idx] - uo[idx] + VSQR *(dt * dt) *(( d_uc_received_v[i] +
                  uc[idx+1] + uc[idx-node_thread_dim] + uc[idx+node_thread_dim] +
                  0.25*(d_uc_received_v[i-1] + uc[idx-node_thread_dim+1] + d_uc_received_v[i+1] +
                  uc[idx+node_thread_dim+1]) - 5 * uc[idx])/(h * h) + f_gpu(pebbles[idx],t));
      } else {
        un[idx] = 2*uc[idx] - uo[idx] + VSQR *(dt * dt) *(( uc[idx-1] +
          uc[idx+1] + uc[idx-node_thread_dim] + uc[idx+node_thread_dim] +
          0.25*(uc[idx-node_thread_dim-1] + uc[idx-node_thread_dim+1] +
            uc[idx+node_thread_dim-1] + uc[idx+node_thread_dim+1])- 5 *
            uc[idx])/(h * h) + f_gpu(pebbles[idx],t));
      }
    }
  } else if(rank == 2) {
    if(i == node_thread_dim-1 || j == 0) {
      un[idx] = 0.;
    }
    else {
      if (i == 0 && j == node_thread_dim-1) {
        // Get from diagonal and horizontal and vertical.
        un[idx] = 2*uc[idx] - uo[idx] + VSQR *(dt * dt) *(( uc[idx-1] +
                  d_uc_received_v[i] + d_uc_received_h[j] + uc[idx+node_thread_dim] +
                  0.25*(d_uc_received_h[j-1] + d_uc_received_d + uc[idx+node_thread_dim-1] +
                  d_uc_received_v[i+1])-5 * uc[idx])/(h * h) + f_gpu(pebbles[idx],t));
      } else if (i == 0 && j != node_thread_dim-1) {
        // Use horizontal.
        un[idx] = 2*uc[idx] - uo[idx] + VSQR *(dt * dt) *(( uc[idx-1] +
                  uc[idx+1] + d_uc_received_h[j] + uc[idx+node_thread_dim] +
                  0.25*(d_uc_received_h[j-1] + d_uc_received_h[j+1] +
                  uc[idx+node_thread_dim-1] + uc[idx+node_thread_dim+1])-5 * uc[idx])/(h * h) + f_gpu(pebbles[idx],t));
      } else if (j == node_thread_dim-1 && i != 0) {
        // Use vertical.
        un[idx] = 2*uc[idx] - uo[idx] + VSQR *(dt * dt) *(( uc[idx-1] +
                  d_uc_received_v[i] + uc[idx-node_thread_dim] + uc[idx+node_thread_dim] +
                  0.25*(d_uc_received_v[i-1] + uc[idx-node_thread_dim+1] + uc[idx+node_thread_dim-1] +
                  d_uc_received_v[i+1])-5 * uc[idx])/(h * h) + f_gpu(pebbles[idx],t));
      } else {
        un[idx] = 2*uc[idx] - uo[idx] + VSQR *(dt * dt) *(( uc[idx-1] +
          uc[idx+1] + uc[idx-node_thread_dim] + uc[idx+node_thread_dim] +
          0.25*(uc[idx-node_thread_dim-1] + uc[idx-node_thread_dim+1] +
            uc[idx+node_thread_dim-1] + uc[idx+node_thread_dim+1])- 5 *
            uc[idx])/(h * h) + f_gpu(pebbles[idx],t));
      }
    }
  } else {
    if(i == node_thread_dim-1 || j == node_thread_dim-1) {
      un[idx] = 0.;
    }
    else {
      if (i == 0 && j == 0) {
        // Get from diagonal and horizontal and vertical.
        un[idx] = 2*uc[idx] - uo[idx] + VSQR *(dt * dt) *(( d_uc_received_v[i] + uc[idx+1] +
                    d_uc_received_h[j] + uc[idx+node_thread_dim] + 0.25*(d_uc_received_d + d_uc_received_h[j+1] + d_uc_received_v[i+1] + uc[idx+node_thread_dim+1])-
                    5 * uc[idx])/(h * h) + f_gpu(pebbles[idx],t));
      } else if (i == 0 && j != 0) {
        // Use horizontal.
        un[idx] = 2*uc[idx] - uo[idx] + VSQR *(dt * dt) *(( uc[idx-1] + uc[idx+1] +
                    d_uc_received_h[j] + uc[idx+node_thread_dim] + 0.25*(d_uc_received_h[j-1] + d_uc_received_h[j+1] + uc[idx+node_thread_dim-1] + uc[idx+node_thread_dim+1])-
                    5 * uc[idx])/(h * h) + f_gpu(pebbles[idx],t));
      } else if (j == 0 && i != 0) {
        // Use vertical.
        un[idx] = 2*uc[idx] - uo[idx] + VSQR *(dt * dt) *(( d_uc_received_v[i] + uc[idx+1] +
                    uc[idx-node_thread_dim] + uc[idx+node_thread_dim] + 0.25*(d_uc_received_v[i-1] + uc[idx-node_thread_dim+1] + d_uc_received_v[i+1] + uc[idx+node_thread_dim+1])-
                    5 * uc[idx])/(h * h) + f_gpu(pebbles[idx],t));
      } else {
        un[idx] = 2*uc[idx] - uo[idx] + VSQR *(dt * dt) *(( uc[idx-1] +
          uc[idx+1] + uc[idx-node_thread_dim] + uc[idx+node_thread_dim] +
          0.25*(uc[idx-node_thread_dim-1] + uc[idx-node_thread_dim+1] +
            uc[idx+node_thread_dim-1] + uc[idx+node_thread_dim+1])- 5 *
            uc[idx])/(h * h) + f_gpu(pebbles[idx],t));
      }
    }
  }
}

int tpdt_gpu(double *t, double dt, double tf)
{
  if((*t) + dt > tf) return 0;
  (*t) = (*t) + dt;
  return 1;
}

// get_boundary_values(d_uo_v, d_uo, n/2 - 1, n);
void get_boundary_values(double *u_v, double *u, int start_point, int n) {
  int current_index = start_point;
  for(int i = 0; i < n/2; i++) {
    u_v[i] = u[current_index];
    current_index += n/2;
  }
}

int run_gpu(double *u, double *u0, double *u1, double *pebbles, int n, double h, double end_time, int nthreads, int argc, char *argv[])
{
  MPI_Init(&argc, &argv);

  int number_of_processes, rank;
  int nBlocks = n/nthreads;

  // Get number of processes as a part of which this program will run
  MPI_Comm_size(MPI_COMM_WORLD, &number_of_processes);

  // Get rank of process in the communicator
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  MPI_Status status;

	cudaEvent_t kstart, kstop;
	float ktime;

	/* HW2: Define your local variables here */
  // Calculate the number of blocks and threads.
  double t = 0.;
  double dt = h / 2.;
  int narea = n * n;

  dim3 no_blocks(nBlocks/2, nBlocks/2);
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
  // Variable for horizontal values received from neighbors.
  double *d_uc_received_h;
  // Variable for vertical values received from neighbors.
  double *d_uc_received_v;
  // Value from diagonal node.
  double d_uc_received_d;

  cudaMalloc((double **)&d_uo, (narea / 4) * sizeof(double));
  cudaMalloc((double **)&d_uc, (narea / 4) * sizeof(double));
  cudaMalloc((double **)&d_un, (narea / 4) * sizeof(double));
  cudaMalloc((double **)&d_pebb, (narea / 4) * sizeof(double));
  // Memory allocation for horizontal values received
  cudaMalloc((double **)&d_uc_received_h, (n / 2) * sizeof(double));
  // Memory allocation for vertical values received
  cudaMalloc((double **)&d_uc_received_v, (n / 2) * sizeof(double));

  // Host copies - for MPI_Send and MPI_Receive and get_boundary_values.
  double *uo;
  double *uc;
  double *un;
  double *pebb;
  // Variable for horizontal values received from neighbors.
  double *uc_received_h;
  // Variable for vertical values received from neighbors.
  double *uc_received_v;
  // Value from diagonal node.
  double uc_received_d;
  // Vertical data calculated by each rank.
  double *uc_v;

  uo = (double *)malloc((narea / 4) * sizeof(double));
  uc = (double *)malloc((narea / 4) * sizeof(double));
  un = (double *)malloc((narea / 4) * sizeof(double));
  pebb = (double *)malloc((narea / 4) * sizeof(double));
  // Memory allocation for horizontal value received
  uc_received_h = (double *)malloc((n / 2) * sizeof(double));
  // Memory allocation for vertical value received
  uc_received_v = (double *)malloc((n / 2) * sizeof(double));
  // Memory allocation for vertical data calculated by each rank.
  uc_v = (double *)malloc((n / 2) * sizeof(double));

  int curr_start;
  // Dividing entire grid into parts according to ranks.
  if (rank == 0) {
    curr_start = 0;
    for (int i = 0; i < n/2; i++, curr_start += n) {
      memcpy(uo + curr_start/2, u0 + curr_start, sizeof(double) * n/2);
      memcpy(uc + curr_start/2, u1 + curr_start, sizeof(double) * n/2);
      memcpy(pebb + curr_start/2, pebbles + curr_start, sizeof(double) * n/2);
    }
  } else if (rank == 1) {
    curr_start = n/2;
    for (int i = 0; i < n/2; i++, curr_start += n) {
      memcpy(uo + (curr_start - n/2) / 2, u0 + curr_start, sizeof(double) * n/2);
      memcpy(uc + (curr_start - n/2) / 2, u1 + curr_start, sizeof(double) * n/2);
      memcpy(pebb + (curr_start - n/2) / 2, pebbles + curr_start, sizeof(double) * n/2);
    }
  } else if (rank == 2) {
    curr_start = n * n / 2;
    for (int i = 0; i < n/2; i++, curr_start += n) {
      memcpy(uo + (curr_start - n*n/2) / 2, u0 + curr_start, sizeof(double) * n/2);
      memcpy(uc + (curr_start - n*n/2) / 2, u1 + curr_start, sizeof(double) * n/2);
      memcpy(pebb + (curr_start - n*n/2) / 2, pebbles + curr_start, sizeof(double) * n/2);
    }
  } else {
    curr_start = n * (n+1) / 2;
    for (int i = 0; i < n/2; i++, curr_start += n) {
      memcpy(uo + (curr_start - n * (n+1) / 2) / 2, u0 + curr_start, sizeof(double) * n/2);
      memcpy(uc + (curr_start - n * (n+1) / 2) / 2, u1 + curr_start, sizeof(double) * n/2);
      memcpy(pebb + (curr_start - n * (n+1) / 2) / 2, pebbles + curr_start, sizeof(double) * n/2);
    }
  }

  /* Start GPU computation timer */
	CUDA_CALL(cudaEventRecord(kstart, 0));
  /* HW2: Add main lake simulation loop here */
  while(1)
  {
    // All the MPI calls between all ranks to get the data needed.
    if (rank == 0) {
      // Vertical send.
      get_boundary_values(uc_v, uc, n/2 - 1, n);
      // Send vertical boundary values to right node (Part of 1st cycle)
      MPI_Send(uc_v, n/2, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
      // Receive horizontal values from below node (Part of 1st cycle)
      MPI_Recv(uc_received_h, n/2, MPI_DOUBLE, 2, 0, MPI_COMM_WORLD, &status);
      // Horizontal boundary values getting sent (Part of 2nd cycle)
      MPI_Send(uc + n/2 * n/2 - n/2, n/2, MPI_DOUBLE, 2, 0, MPI_COMM_WORLD);
      // Receive vertical boundary values from right node (Part of 2nd cycle)
      MPI_Recv(uc_received_v, n/2, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, &status);
      // Send diagonal value
      double uc_diag = uc[(n/2 * n/2) - 1];
      MPI_Send(&uc_diag, 1, MPI_DOUBLE, 3, 0, MPI_COMM_WORLD);
      // Receive diagonal value
      MPI_Recv(&uc_received_d, 1, MPI_DOUBLE, 3, 0, MPI_COMM_WORLD, &status);
    } else if (rank == 1) {
      // Receive vertical boundary values from left node (Part of 1st cycle)
      MPI_Recv(uc_received_v, n/2, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
      // Send Horizontal boundary values to node below (Part of 1st cycle)
      MPI_Send(uc + n/2 * n/2 - n/2, n/2, MPI_DOUBLE, 3, 0, MPI_COMM_WORLD);
      // Receive horizontal boundary values from below node (Part of 2nd cycle)
      MPI_Recv(uc_received_h, n/2, MPI_DOUBLE, 3, 0, MPI_COMM_WORLD, &status);
      // Calculate and send vertical boundary values to left node (Part of 2nd cycle)
      get_boundary_values(uc_v, uc, 0, n);
      MPI_Send(uc_v, n/2, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
      // Receive diagonal values
      MPI_Recv(&uc_received_d, 1, MPI_DOUBLE, 2, 0, MPI_COMM_WORLD, &status);
      // Send diagonal value
      double uc_diag = uc[(n/2 * n/2) - n/2];
      MPI_Send(&uc_diag, 1, MPI_DOUBLE, 2, 0, MPI_COMM_WORLD);
    } else if (rank == 2) {
      // Receive vertical boundary values from right node (Part of 1st cycle)
      MPI_Recv(uc_received_v, n/2, MPI_DOUBLE, 3, 0, MPI_COMM_WORLD, &status);
      // Send horizontal values to above node (Part of 1st cycle)
      MPI_Send(uc, n/2, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
      // Receive horizontal boundary values from above node (Part of 2nd cycle)
      MPI_Recv(uc_received_h, n/2, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
      // Calculate and send vertical boundary values to right node (Part of 2nd cycle)
      get_boundary_values(uc_v, uc, n/2 - 1, n);
      // printf("After get boundary with rank - %d\n", rank);
      MPI_Send(uc_v, n/2, MPI_DOUBLE, 3, 0, MPI_COMM_WORLD);
      // Send diagonal value
      double uc_diag = uc[n/2 - 1];
      MPI_Send(&uc_diag, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
      // Receive diagonal values
      MPI_Recv(&uc_received_d, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, &status);
    } else {
      // Receive horizontal boundary values from node above (Part of 1st cycle)
      MPI_Recv(uc_received_h, n/2, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, &status);
      // Calculate vertical values and then send to left node (Part of 1st cycle)
      get_boundary_values(uc_v, uc, 0, n);
      MPI_Send(uc_v, n/2, MPI_DOUBLE, 2, 0, MPI_COMM_WORLD);
      // Receive vertical boundary values from left node (Part of 2nd cycle)
      MPI_Recv(uc_received_v, n/2, MPI_DOUBLE, 2, 0, MPI_COMM_WORLD, &status);
      // Send horizontal boundary values to above node (Part of 2nd cycle)
      MPI_Send(uc, n/2, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
      // Receive diagonal value
      MPI_Recv(&uc_received_d, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
      // Send diagonal value
      double uc_diag = uc[0];
      MPI_Send(&uc_diag, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

    // Do cudaMemcpy for all host copies to device copies before calling kernel function 'evolve()'
    cudaMemcpy(d_uo, uo, (narea / 4) * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_uc, uc, (narea / 4) * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_un, un, (narea / 4) * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pebb, pebb, (narea / 4) * sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(d_uc_received_h, uc_received_h, (n / 2) * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_uc_received_v, uc_received_v, (n / 2) * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(&d_uc_received_d, &uc_received_d, sizeof(double) * 1, cudaMemcpyHostToDevice);

    evolve_gpu<<<no_blocks, threads_per_block>>>(d_un, d_uc, d_uo, d_pebb, n/2, h, dt, t, rank, d_uc_received_h, d_uc_received_v, d_uc_received_d);

    cudaMemcpy(uo, d_uc, sizeof(double) * narea/4, cudaMemcpyDeviceToHost);
    cudaMemcpy(uc, d_un, sizeof(double) * narea/4, cudaMemcpyDeviceToHost);

    if(!tpdt_gpu(&t,dt,end_time)) break;
  }

  /* Copy back results to host */
  cudaMemcpy(u, d_un, sizeof(double) * narea/4, cudaMemcpyDeviceToHost);

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
  cudaFree(d_uc_received_h);
  cudaFree(d_uc_received_v);

  free(uo);
  free(uc);
  free(un);
  free(pebb);
  free(uc_received_h);
  free(uc_received_v);
  free(uc_v);
	/* timer cleanup */
	CUDA_CALL(cudaEventDestroy(kstart));
	CUDA_CALL(cudaEventDestroy(kstop));

  // Graceful exit
  MPI_Finalize();

  return rank;
}
