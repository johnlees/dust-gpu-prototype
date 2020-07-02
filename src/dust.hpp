#ifndef DUST_DUST_HPP
#define DUST_DUST_HPP

#include "rng.hpp"

#include <utility>
#ifdef _OPENMP
#if _OPENMP >= 201511
#define OPENMP_HAS_MONOTONIC 1
#endif
#include <omp.h>
#endif

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

template <typename T>
class Particle {
public:
  typedef typename T::init_t init_t;
  typedef typename T::int_t int_t;
  typedef typename T::real_t real_t;
  typedef typename dust::RNG<real_t, int_t> rng_t;

  Particle(init_t data, size_t step) :
    _model(data),
    _step(step),
    _y(_model.initial(_step)),
    _y_swap(_model.size()) {
      _y_device = _y;
      _y_swap_device = _y_swap;
  }

  real_t * y_addr() { return thrust::raw_pointer_cast(&_y_device[0]) };
  real_t * y_swap_addr() { return thrust::raw_pointer_cast(&_y_swap_device[0]) };

  void state(const std::vector<size_t>& index_y,
             typename std::vector<real_t>::iterator end_state) const {
    _y = _y_device;
    for (size_t i = 0; i < index_y.size(); ++i) {
      *(end_state + i) = _y[index_y[i]];
    }
  }

  void state_full(typename std::vector<real_t>::iterator end_state) const {
    _y = _y_device;
    for (size_t i = 0; i < _y.size(); ++i) {
      *(end_state + i) = _y[i];
    }
  }

  size_t size() const {
    return _y.size();
  }

  size_t step() const {
    return _step;
  }

  void swap() {
    _y = _y_device;
    _y_swap = _y_swap_device;
    std::swap(_y, _y_swap);
  }

  void update(const Particle<T> other) {
    _y_swap = other._y;
    _y_swap_device = _y_swap;
  }

private:
  T _model;
  size_t _step;

  std::vector<real_t> _y;
  std::vector<real_t> _y_swap;
  thrust::device_vector<real_t> _y_device;
  thrust::device_vector<real_t> _y_swap_device;
};

template <typename T>
class Dust {
public:
  typedef typename T::init_t init_t;
  typedef typename T::int_t int_t;
  typedef typename T::real_t real_t;
  typedef typename dust::RNG<real_t, int_t> rng_t;

  Dust(const init_t data, const size_t step,
       const std::vector<size_t> index_y,
       const size_t n_particles, const size_t n_threads,
       const size_t n_generators, const size_t seed) :
    _index_y(index_y),
    _n_threads(n_threads),
    _rng(n_generators, seed) {
    std::vector<real_t*> y_ptrs;
    std::vector<real_t*> y_swap_ptrs;
    for (size_t i = 0; i < n_particles; ++i) {
      _particles.push_back(Particle<T>(data, step));
      y_ptrs.push_back(_particles.y_addr());
      y_swap_ptrs.push_back(_particles.y_swap_addr());
    }
    _model = new T(data, step);
    cudaMallocManaged((void** )&_model_device, sizeof(T));
    cudaMemcpy(_model_device, _model, sizeof(T),
	              cudaMemcpyHostToDevice);

    cudaMallocManaged((void** )&_particle_y_addrs, y_ptrs.size() * sizeof(real_t*));
    cudaMemcpy(_particle_y_addrs, y_ptrs.data(), y_ptrs.size() * sizeof(real_t*),
	              cudaMemcpyHostToDevice);
    cudaMallocManaged((void** )&_particle_y_swap_addrs, y_swap_ptrs.size() * sizeof(real_t*));
    cudaMemcpy(_particle_y_addrs, y_swap_ptrs.data(), y_swap_ptrs.size() * sizeof(real_t*),
	              cudaMemcpyHostToDevice)
  }

  ~Dust() {
    delete _model;
    cudaFree(_model_device);
    cudaFree(_particle_y_addrs);
    cudaFree(_particle_y_swap_addrs);
  }

  void reset(const init_t data, const size_t step) {
    size_t n_particles = _particles.size();
    _particles.clear();
    for (size_t i = 0; i < n_particles; ++i) {
      _particles.push_back(Particle<T>(data, step));
    }
  }

  __global__
  void run_particles(T* _model_device,
                    real_t** particle_y,
                    real_t** particle_y_swap,
                    size_t n_particles,
                    size_t step,
                    size_t step_end) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (long long p_idx = index; p_idx < n_particles; p_idx += stride) {
      size_t curr_step = step
      while (curr_step < step_end) {
        _model_device->update(curr_step, particle_y[p_idx], rng, particle_y_swap[p_idx]);
        curr_step++;
        for (int i = 0; i < y_len; i++) {
          real_t tmp = particle_y[p_idx][i];
          particle_y[p_idx][i] = particle_y_swap[p_idx][i];
          particle_y_swap[p_idx][i] = tmp;
        }
      }
    }
  }

  void run(const size_t step_end) {
    const size_t blockSize = 32; // Check later
    const size_t blockCount = (_particles.size() + blockSize - 1) / blockSize;
    run_particles<<<blockCount, blockSize>>(_model_device,
                                            _particle_y_addrs,
                                            _particle_y_swap_addrs,
                                            _particles.size(),
                                            this->step(),
                                            step_end);
    // TODO: write step end back to particles
  }

  void state(std::vector<real_t>& end_state) const {
    #pragma omp parallel for schedule(static) num_threads(_n_threads)
    for (size_t i = 0; i < _particles.size(); ++i) {
      _particles[i].state(_index_y, end_state.begin() + i * _index_y.size());
    }
  }

  void state(std::vector<size_t> index_y,
             std::vector<real_t>& end_state) const {
    #pragma omp parallel for schedule(static) num_threads(_n_threads)
    for (size_t i = 0; i < _particles.size(); ++i) {
      _particles[i].state(index_y, end_state.begin() + i * index_y.size());
    }
  }

  void state_full(std::vector<real_t>& end_state) const {
    const size_t n = n_state_full();
    #pragma omp parallel for schedule(static) num_threads(_n_threads)
    for (size_t i = 0; i < _particles.size(); ++i) {
      _particles[i].state_full(end_state.begin() + i * n);
    }
  }

  // There are two obvious ways of reordering; we can construct a
  // completely new set of particles, like
  //
  //   std::vector<Particle<T>> next;
  //   for (auto const& i: index) {
  //     next.push_back(_particles[i]);
  //   }
  //   _particles = next;
  //
  // but this seems like a lot of churn.  The other way is to treat it
  // like a slightly weird state update where we swap around the
  // contents of the particle state (uses the update() and swap()
  // methods on particles).
  void reorder(const std::vector<size_t>& index) {
    for (size_t i = 0; i < _particles.size(); ++i) {
      size_t j = index[i];
      _particles[i].update(_particles[j]);
    }
    for (auto& p : _particles) {
      p.swap();
    }
  }

  size_t n_particles() const {
    return _particles.size();
  }
  size_t n_state() const {
    return _index_y.size();
  }
  size_t n_state_full() const {
    return _particles.front().size();
  }
  size_t step() const {
    return _particles.front().step();
  }

private:
  const std::vector<size_t> _index_y;
  const size_t _n_threads;
  dust::pRNG<real_t, int_t> _rng;
  std::vector<Particle<T>> _particles;

  T* _model;
  T* _model_device;
  real_t** _particle_y_addrs;
  real_t** _particle_y_swap_addrs;

  // For 10 particles, 4 generators and 1, 2, 4 threads we want this:
  //
  // i:  0 1 2 3 4 5 6 7 8 9
  // g:  0 1 2 3 0 1 2 3 0 1 - rng used for the iteration
  // t1: 0 0 0 0 0 0 0 0 0 0 - thread index that executes each with 1 thread
  // t2: 0 1 0 1 0 1 0 1 0 1 - ...with 2
  // t4: 0 1 2 3 0 1 2 3 0 1 - ...with 4
  //
  // So with
  // - 1 thread: 0: (0 1 2 3)
  // - 2 threads 0: (0 2), 1: (1 3)
  // - 4 threads 0: (0), 1: (1), 2: (2), 3: (3)
  //
  // So the rng number can be picked up directly by doing
  //
  //   i % _rng.size()
  //
  // though this relies on the openmp scheduler, which technically I
  // think we should not be doing. We could derive it from the thread
  // index to provide a set of allowable rngs but this will be harder
  // to get deterministic.
  //
  // I'm not convinced that this will always do the Right Thing with
  // loop leftovers either.
  rng_t& pick_generator(const size_t i) {
    return _rng(i % _rng.size());
  }
};

#endif