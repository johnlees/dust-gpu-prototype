# dust-gpu-prototype
Prototype code for CUDA version of dust

Runs a stochastic SIRS model with many repeats (particles).

## Compile

Run `make`. 

For debugging `make DEBUG=1`. For profiling `make PROFILE=1`.

`make clean && make` to recompile.

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
```
