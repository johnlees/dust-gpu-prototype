# dust-gpu-prototype
Prototype code for CUDA version of dust

Runs a stochastic SIRS model with many repeats (particles).

## Cluster setup

After cloning the repo, launch an interactive session requesting 1 GPU and 2 CPUs:
```
srun --ntasks=5 --nodes=1 --cpus-per-task=2 --partition=batch --time=4:00:00 --gres=gpu:1 --pty /bin/bash
```

Set up the compilers:
```
module load cuda nvcompilers
```

## Compile

`cd src/` then run `make`.

For debugging `make DEBUG=1`. For profiling `make PROFILE=1`.

`make clean` to recompile.

## Run

```
./dust_test
-a rate of recovered to susceptible (float; default = 0.1)
-b rate of infection (float; default = 0.2)
-g rate of recovery (float; default = 0.1)
-S number of initial susceptible (int; default = 1000)
-I number of initial infected (int; default = 10)
-R number of initial recovered (int; default = 0)
-p number of particles to run (int; default = 100)
-s number of time steps to simulate each particle forward (int; default = 10000)
-t print current populations every t time steps. 0->do not print (int; default = 0)
```

## Validation

Simulation from R on the CPU (in float mode)

```r
pkgload::load_all("tmp")
obj <- sireinfect$new(list(I_ini = 10), 0, n_particles = 5, seed = 1)
obj$run(10)
#>      [,1] [,2] [,3] [,4] [,5]
#> [1,]  999  958  981  983  978
#> [2,]    6   37   20   18   21
#> [3,]    5   15    9    9   11
```

Comparison on the CPU

```
./src/dust_test -p 5 -t 10 -s 11
Step: 10
P	S	I	R
0	999	6	5
1	958	37	15
2	981	20	9
3	983	18	9
4	978	21	11
elapsed time: 0.00094882s
```
