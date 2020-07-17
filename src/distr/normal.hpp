#ifndef DUST_DISTR_NORMAL_HPP
#define DUST_DISTR_NORMAL_HPP

#include <cmath>

namespace dust {
namespace distr {

__device__
inline void BoxMullerFloat(uint64_t x0, uint64_t x1, float* f0, float* f1) {
  // This function implements the Box-Muller transform:
  // http://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform#Basic_form
  // Do not send a really small number to log().
  // We cannot mark "epsilon" as "static const" because NVCC would complain
  const float epsilon = 1.0e-7f;
  float u1 = __ull2float_rz(x0);
  if (u1 < epsilon) {
    u1 = epsilon;
  }
  const float v1 = 2.0f * M_PI * __ull2float_rz(x1);
  const float u2 = sqrtf(-2.0f * logf(u1));
  sincosf(v1, f0, f1);
  *f0 *= u2;
  *f1 *= u2;
}

__device__
inline void BoxMullerDouble(uint64_t x0, uint64_t x1, double* d0,
                     double* d1) {
  // This function implements the Box-Muller transform:
  // http://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform#Basic_form
  // Do not send a really small number to log().
  // We cannot mark "epsilon" as "static const" because NVCC would complain
  const double epsilon = 1.0e-7;
  double u1 = __ull2double_rz(x0);
  if (u1 < epsilon) {
    u1 = epsilon;
  }
  const double v1 = 2 * M_PI * __ull2double_rz(x1);;
  const double u2 = sqrt(-2.0 * log(u1));
  sincos(v1, d0, d1);
  *d0 *= u2;
  *d1 *= u2;
}

class NormalDistribution<float> {
 public:
  __device__
  NormalDistribution() : _buffered(false) {}

  __device__
  float operator()(uint64_t* rng_state) {
    if (buffered) {
      buffered = false;
      return result[1];
    } else {
      uint64_t u0 = gen_rand(rng_state);
      uint64_t u1 = gen_rand(rng_state);
      BoxMullerFloat(u0, u1, &result[0], &result[1]);
      buffered = true;
      return result[0];
    }
  }

  private:
    bool _buffered;
    float result[2];
};

class NormalDistribution<double> {
 public:
  __device__
  NormalDistribution() : _buffered(false) {}

  __device__
  double operator()(uint64_t* rng_state) {
    if (buffered) {
      buffered = false;
      return result[1];
    } else {
      uint64_t u0 = gen_rand(rng_state);
      uint64_t u1 = gen_rand(rng_state);
      BoxMullerDouble(u0, u1, &result[0], &result[1]);
      buffered = true;
      return result[0];
    }
  }

  private:
    bool _buffered;
    double result[2];
};

}
}

#endif
