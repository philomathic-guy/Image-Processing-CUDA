/*
Group info:
nphabia Niklesh Phabiani
rtnaik	Rohit Naik
anjain2 Akshay Narendra Jain
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#define _USE_MATH_DEFINES

#define XMIN 0.0
#define XMAX 1.0
#define YMIN 0.0
#define YMAX 1.0

// Number of points to be considered around a point.
#define STENCIL 9

#define MAX_PSZ 10
#define TSCALE 1.0
#define VSQR 0.1

void init(double *u, double *pebbles, int n);
void print_heatmap(const char *filename, double *u, int n, double h);
void init_pebbles(double *p, int pn, int n);

extern int run_gpu(double *u, double *u0, double *u1, double *pebbles, int n, double h, double end_time, int nthreads, int argc, char *argv[]);

int main(int argc, char *argv[])
{
  if(argc != 5)
  {
    printf("Usage: %s npoints npebs time_finish nthreads \n",argv[0]);
    return 0;
  }

  int     npoints   = atoi(argv[1]);
  int     npebs     = atoi(argv[2]);
  double  end_time  = (double)atof(argv[3]);
  int     nthreads  = atoi(argv[4]);
  int 	  narea	    = npoints * npoints;

  double *u_i0, *u_i1;
  double *u_gpu, *pebs;
  double h;

  double elapsed_gpu;
  struct timeval gpu_start, gpu_end;

  u_i0 = (double*)malloc(sizeof(double) * narea);
  u_i1 = (double*)malloc(sizeof(double) * narea);
  pebs = (double*)malloc(sizeof(double) * narea);

  // Since each node will create its own dat file in the end.
  u_gpu = (double*)malloc(sizeof(double) * narea/4);

  printf("Running %s with (%d x %d) grid, until %f, with %d threads\n", argv[0], npoints, npoints, end_time, nthreads);

  init_pebbles(pebs, npebs, npoints);
  init(u_i0, pebs, npoints);
  init(u_i1, pebs, npoints);

  // For initial image, keep h same as we have not split the initial image into 4 parts.
  h = (XMAX - XMIN)/npoints;

  print_heatmap("lake_i.dat", u_i0, npoints, h);

  // Since we have reduced the size of u_gpu to the part of the image on each node, we scale the image back to full-size.
  h = (XMAX - XMIN)/(npoints/2);

  gettimeofday(&gpu_start, NULL);
  int rank = run_gpu(u_gpu, u_i0, u_i1, pebs, npoints, h, end_time, nthreads, argc, argv);
  gettimeofday(&gpu_end, NULL);

  elapsed_gpu = ((gpu_end.tv_sec + gpu_end.tv_usec * 1e-6)-(
                  gpu_start.tv_sec + gpu_start.tv_usec * 1e-6));
  printf("GPU took %f seconds\n", elapsed_gpu);

  char filename[50];
  sprintf(filename, "lake_f_%d.dat", rank);
  print_heatmap(filename, u_gpu, npoints/2, h);

  free(u_i0);
  free(u_i1);
  free(pebs);
  free(u_gpu);

  return 1;
}

void init_pebbles(double *p, int pn, int n)
{
  int i, j, k, idx;
  int sz;

  srand( time(NULL) );
  memset(p, 0, sizeof(double) * n * n);

  for( k = 0; k < pn ; k++ )
  {
    i = rand() % (n - 4) + 2;
    j = rand() % (n - 4) + 2;
    sz = rand() % MAX_PSZ;
    idx = j + i * n;
    p[idx] = (double) sz;
  }
}

double f(double p, double t)
{
  return -expf(-TSCALE * t) * p;
}

void init(double *u, double *pebbles, int n)
{
  int i, j, idx;

  for(i = 0; i < n ; i++)
  {
    for(j = 0; j < n ; j++)
    {
      idx = j + i * n;
      u[idx] = f(pebbles[idx], 0.0);
    }
  }
}

void print_heatmap(const char *filename, double *u, int n, double h)
{
  int i, j, idx;

  FILE *fp = fopen(filename, "w");

  for( i = 0; i < n; i++ )
  {
    for( j = 0; j < n; j++ )
    {
      idx = j + i * n;
      fprintf(fp, "%f %f %f\n", i*h, j*h, u[idx]);
    }
  }

  fclose(fp);
}
