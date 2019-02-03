Script created to run for all sizes.

V1:
Both 5-point and 9-point stencil are used for discretizing the wave equation in the spatial domain. Here, both are going deeper into the layers
to see the spread of ripples over time. But we know that ripples spread out and then diminish, and this is how evolve works. So, from our experiments
we did observe that the 5-point stencil took longer time to evolve smoothly before starting to diminish (around 1 second). On the other hand, the
9-point stencil had a nice smooth evolve in around 0.6 seconds whereas on running the 9 point stencil for 1 second, it led to a blur version indicating
the diminishing of the ripples. Also, 9 point is observed to be more accurate since it takes into account the impact of all the immediate neighbors,
thereby producing a more smoother plot. But since 9 point involves more computations, it will take more time to perform a single iteration evolve but with a
more accurate result.

V2:
bash-4.2$ ./lake_v2.sh

Running ./lake with (16 x 16) grid, until 0.600000, with 8 threads
CPU took 0.000707 seconds
GPU computation: 2.901280 msec
GPU took 0.469634 seconds

Running ./lake with (32 x 32) grid, until 0.600000, with 8 threads
CPU took 0.002371 seconds
GPU computation: 0.844768 msec
GPU took 0.301947 seconds

Running ./lake with (64 x 64) grid, until 0.600000, with 8 threads
CPU took 0.020183 seconds
GPU computation: 1.659616 msec
GPU took 0.303288 seconds

Running ./lake with (128 x 128) grid, until 0.600000, with 8 threads
CPU took 0.162070 seconds
GPU computation: 4.444384 msec
GPU took 0.316021 seconds

Running ./lake with (256 x 256) grid, until 0.600000, with 8 threads
CPU took 1.314868 seconds
GPU computation: 27.888479 msec
GPU took 0.340243 seconds

Running ./lake with (512 x 512) grid, until 0.600000, with 8 threads
CPU took 12.325901 seconds
GPU computation: 208.160614 msec
GPU took 0.511493 seconds

Running ./lake with (1024 x 1024) grid, until 0.600000, with 8 threads
CPU took 105.858671 seconds
GPU computation: 1435.478638 msec
GPU took 1.832637 seconds

It can be seen that for smaller grid sizes CPU performs better than GPU because of the overhead incurred by the GPU kernel call invocation, and
not having many computations to be performed on a large number of threads, which can be provided by the GPU. But, as the grid size increases, more
work is involved and the large number of threads doing simultaneous processing perform this work faster than CPU,
thereby compensating for the overhead incurred for the GPU kernel call invocation. The GPU implementation does not scale well till 128 X 128 grid size,
but it can seen that from the grid size of 256 onwards, GPU implementation starts performing better than CPU. The time difference keeps increasing
with an increase in the grid size. Hence, more work being available uses the provisioning done by GPU to use a large number of threads to do the work
or chunks of work simultaneously.

We integrated CUDA with MPI by dividing the grid into 4 quadrants.
  ----------------
 |	  	  | 		  |
 |   1    |   3	  |
 |	  	  |       |
  ----------------
 |	  	  |       |
 |   0    |   2	  |
 |	  	  |       |
  ----------------
NOTE: While running, we get a message - "BAD TERMINATION OF ONE OF YOUR APPLICATION PROCESSES".
We think this is because of some node issue because the output png files that we get when joined according to the above configuration
will give us the correct answer for the entire lake (wrt the pebble positions).

Problems that we encountered were about passing data from the neighboring rank nodes to the current node for calculation of each point in the current node.
While doing this you have to do MPI calls between nodes. The problem that we faced here and realized our mistake
was that we cannot pass device memory through MPI calls to nodes directly. We have to pass host memory and then copy it to device at each node.
