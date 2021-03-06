#ifndef DUST_XOSHIRO_HPP
#define DUST_XOSHIRO_HPP

#include <cstdint>
#include <vector>
#include <limits>
#include <stdio.h>

// This is derived from http://prng.di.unimi.it/xoshiro256starstar.c
// and http://prng.di.unimi.it/splitmix64.c, copies of which are
// included in the package (in inst/rng in the source package). The
// original code is CC0 licenced but was written by David Blackman and
// Sebastiano Vigna.
//
// MD5 (splitmix64.c) = 7e38529aa7bb36624ae4a9d6808e7e3f
// MD5 (xoshiro256starstar.c) = 05f9ecd49bbed98304d982313c91d0f6

#define XOSHIRO_WIDTH 4

namespace dust {

__host__ __device__
static inline uint64_t rotl(const uint64_t x, int k) {
	return (x << k) | (x >> (64 - k));
}

__host__ __device__
inline uint64_t gen_rand(uint64_t * state) {
  const uint64_t result = rotl(state[1] * 5, 7) * 9;
  //printf("r:%lu s:%lu %lu %lu %lu\n", result, state[0], state[1], state[2], state[3]);

  const uint64_t t = state[1] << 17;

  state[2] ^= state[0];
  state[3] ^= state[1];
  state[1] ^= state[2];
  state[0] ^= state[3];

  state[2] ^= t;

  state[3] = rotl(state[3], 45);

  return result;
}

class Xoshiro {
public:
  // Definitions to satisfy interface of URNG in C++11
  typedef uint64_t result_type;
  __host__
  static constexpr uint64_t min() {
    return std::numeric_limits<uint64_t>::min();
  }
  __host__
  static constexpr uint64_t max() {
    return std::numeric_limits<uint64_t>::max();
  }

  __host__
  uint64_t operator()() { return(gen_rand(_state)); };

  /*
  __host__
  T unif_rand() {
    static std::uniform_real_distribution<T> unif_dist(0, 1);
    return unif_dist(*this);
  }
  */

  __host__
  Xoshiro(uint64_t seed);

  // Change internal state
  __host__
  void set_seed(uint64_t seed);
  __host__
  void jump();
  __host__
  void long_jump();

  // Get state
  __host__
  uint64_t* get_rng_state() { return _state; }

private:
  static uint64_t splitmix64(uint64_t seed);
  uint64_t _state[XOSHIRO_WIDTH];
};

__device__
inline double device_unif_rand(uint64_t * state) {
  double rand =
    (__ddiv_rn(__ull2double_rn(gen_rand(state)),
               __ull2double_rn(UINT64_MAX)));
  //printf("r:%f s:%lu %lu %lu %lu\n", rand, state[0], state[1], state[2], state[3]);
  return rand;
}

__device__
inline float device_unif_randf(uint64_t * state) {
  return(__double2float_rn(device_unif_rand(state)));
}

__host__
inline uint64_t Xoshiro::splitmix64(uint64_t seed) {
  uint64_t z = (seed += 0x9e3779b97f4a7c15);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
  z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
  return z ^ (z >> 31);
}

__host__
inline Xoshiro::Xoshiro(uint64_t seed) {
  set_seed(seed);
}

__host__
inline void Xoshiro::set_seed(uint64_t seed) {
  // normal brain: for i in 1:4
  // advanced brain: -funroll-loops
  // galaxy brain:
  _state[0] = splitmix64(seed);
  _state[1] = splitmix64(_state[0]);
  _state[2] = splitmix64(_state[1]);
  _state[3] = splitmix64(_state[2]);
}

/* This is the jump function for the generator. It is equivalent
   to 2^128 calls to next(); it can be used to generate 2^128
   non-overlapping subsequences for parallel computations. */
__host__
inline void Xoshiro::jump() {
  static const uint64_t JUMP[] = \
    { 0x180ec6d33cfd0aba, 0xd5a61266f0c9392c, 0xa9582618e03fc9aa, 0x39abdc4529b1661c };

  uint64_t s0 = 0;
  uint64_t s1 = 0;
  uint64_t s2 = 0;
  uint64_t s3 = 0;
  for (long unsigned int i = 0; i < sizeof JUMP / sizeof *JUMP; i++) {
    for (int b = 0; b < 64; b++) {
      if (JUMP[i] & UINT64_C(1) << b) {
        s0 ^= _state[0];
        s1 ^= _state[1];
        s2 ^= _state[2];
        s3 ^= _state[3];
      }
      this->operator()();
    }
  }

  _state[0] = s0;
  _state[1] = s1;
  _state[2] = s2;
  _state[3] = s3;
}

/* This is the long-jump function for the generator. It is equivalent to
   2^192 calls to next(); it can be used to generate 2^64 starting points,
   from each of which jump() will generate 2^64 non-overlapping
   subsequences for parallel distributed computations. */
__host__
inline void Xoshiro::long_jump() {
  static const uint64_t LONG_JUMP[] = \
    { 0x76e15d3efefdcbbf, 0xc5004e441c522fb3, 0x77710069854ee241, 0x39109bb02acbe635 };

  uint64_t s0 = 0;
  uint64_t s1 = 0;
  uint64_t s2 = 0;
  uint64_t s3 = 0;
  for (long unsigned int i = 0; i < sizeof LONG_JUMP / sizeof *LONG_JUMP; i++) {
    for (int b = 0; b < 64; b++) {
      if (LONG_JUMP[i] & UINT64_C(1) << b) {
        s0 ^= _state[0];
        s1 ^= _state[1];
        s2 ^= _state[2];
        s3 ^= _state[3];
      }
      this->operator()();
    }
  }

  _state[0] = s0;
  _state[1] = s1;
  _state[2] = s2;
  _state[3] = s3;
}

}

#endif
