#include <dust/dust.hpp>
#include <dust/interface.hpp>

// Initial version denerated by odin.dust (version 0.0.3)
class sireinfect {
public:
  typedef int int_t;
  typedef float real_t;
  struct init_t {
    real_t alpha;
    real_t beta;
    real_t gamma;
    real_t I_ini;
    real_t initial_I;
    real_t initial_R;
    real_t initial_S;
    real_t p_IR;
    real_t p_RS;
    real_t S_ini;
  };
  sireinfect(const init_t& data): internal(data) {
  }
  size_t size() {
    return 3;
  }
  std::vector<real_t> initial(size_t step) {
    std::vector<real_t> state(3);
    state[0] = internal.initial_S;
    state[1] = internal.initial_I;
    state[2] = internal.initial_R;
    return state;
  }
  void update(size_t step, const std::vector<real_t>& state, dust::RNG<real_t, int_t>& rng, std::vector<real_t>& state_next) {
    const real_t S = state[0];
    const real_t I = state[1];
    const real_t R = state[2];
    real_t N = S + I + R;
    real_t n_IR = rng.rbinom(std::round(I), internal.p_IR);
    real_t n_RS = rng.rbinom(std::round(R), internal.p_RS);
    real_t p_SI = 1 - std::exp(- internal.beta * I / (real_t) N);
    real_t n_SI = rng.rbinom(std::round(S), p_SI);
    state_next[2] = R + n_IR - n_RS;
    state_next[1] = I + n_SI - n_IR;
    state_next[0] = S - n_SI + n_RS;
  }
private:
  init_t internal;
};
#include <array>
#include <Rcpp.h>

// These would be nice to make constexpr but the way that NA values
// are defined in R's include files do not allow it.
template <typename T>
inline T na_value();

template <>
inline int na_value<int>() {
  return NA_INTEGER;
}

template <>
inline double na_value<double>() {
  return NA_REAL;
}

template <typename T>
inline bool is_na(T x);

template <>
inline bool is_na(int x) {
  return Rcpp::traits::is_na<INTSXP>(x);
}

template <>
inline bool is_na(double x) {
  return Rcpp::traits::is_na<REALSXP>(x);
}

inline size_t object_length(Rcpp::RObject x) {
  // This is not lovely but Rcpp make it really hard to know what a
  // better way is as there seems to be no usable documentation still.
  return ::Rf_xlength(x);
}

template <typename T>
void user_check_value(T value, const char *name, T min, T max) {
  if (ISNA(value)) {
    Rcpp::stop("'%s' must not be NA", name);
  }
  if (min != na_value<T>() && value < min) {
    Rcpp::stop("Expected '%s' to be at least %g", name, (double) min);
  }
  if (max != na_value<T>() && value > max) {
    Rcpp::stop("Expected '%s' to be at most %g", name, (double) max);
  }
}

template <typename T>
void user_check_array_value(const std::vector<T>& value, const char *name,
                            T min, T max) {
  for (auto& x : value) {
    user_check_value(x, name, min, max);
  }
}

inline size_t user_get_array_rank(Rcpp::RObject x) {
  if (x.hasAttribute("dim")) {
    Rcpp::IntegerVector dim = x.attr("dim");
    return dim.size();
  } else {
    // This is not actually super correct
    return 1;
  }
}

template <size_t N>
void user_check_array_rank(Rcpp::RObject x, const char *name) {
  size_t rank = user_get_array_rank(x);
  if (rank != N) {
    if (N == 1) {
      Rcpp::stop("Expected a vector for '%s'", name);
    } else if (N == 2) {
      Rcpp::stop("Expected a matrix for '%s'", name);
    } else {
      Rcpp::stop("Expected an array of rank %d for '%s'", N, name);
    }
  }
}

template <size_t N>
void user_check_array_dim(Rcpp::RObject x, const char *name,
                          const std::array<int, N>& dim_expected) {
  Rcpp::IntegerVector dim = x.attr("dim");
  for (size_t i = 0; i < N; ++i) {
    if (dim[i] != dim_expected[i]) {
      Rf_error("Incorrect size of dimension %d of '%s' (expected %d)",
               i + 1, name, dim_expected[i]);
    }
  }
}

template <>
inline void user_check_array_dim<1>(Rcpp::RObject x, const char *name,
                                    const std::array<int, 1>& dim_expected) {
  if ((int)object_length(x) != dim_expected[0]) {
    Rcpp::stop("Expected length %d value for '%s'", dim_expected[0], name);
  }
}

template <size_t N>
void user_set_array_dim(Rcpp::RObject x, const char *name,
                        std::array<int, N>& dim) {
  Rcpp::IntegerVector dim_given = x.attr("dim");
  std::copy(dim_given.begin(), dim_given.end(), dim.begin());
}

template <>
inline void user_set_array_dim<1>(Rcpp::RObject x, const char *name,
                                  std::array<int, 1>& dim) {
  dim[0] = object_length(x);
}

template <typename T>
T user_get_scalar(Rcpp::List user, const char *name,
                  const T previous, T min, T max) {
  T ret = previous;
  if (user.containsElementNamed(name)) {
    Rcpp::RObject x = user[name];
    if (object_length(x) != 1) {
      Rcpp::stop("Expected a scalar numeric for '%s'", name);
    }
    // TODO: when we're getting out an integer this is a bit too relaxed
    if (Rcpp::is<Rcpp::NumericVector>(x)) {
      ret = Rcpp::as<T>(x);
    } else if (Rcpp::is<Rcpp::IntegerVector>(x)) {
      ret = Rcpp::as<T>(x);
    } else {
      Rcpp::stop("Expected a numeric value for %s", name);
    }
  }

  if (is_na(ret)) {
    Rcpp::stop("Expected a value for '%s'", name);
  }
  user_check_value<T>(ret, name, min, max);
  return ret;
}

// This is not actually really enough to work generally as there's an
// issue with what to do with checking previous, min and max against
// NA_REAL -- which is not going to be the correct value for float
// rather than double.  Further, this is not extendable to the vector
// cases because we hit issues around partial template specification.
//
// We can make the latter go away by replacing std::array<T, N> with
// std::vector<T> - the cost is not great.  But the NA issues remain
// and will require further thought. However, this template
// specialisation and the tests that use it ensure that the core code
// generation is at least compatible with floats.
//
// See #6
template <>
inline float user_get_scalar<float>(Rcpp::List user, const char *name,
                                    const float previous, float min, float max) {
  double value = user_get_scalar<double>(user, name, previous, min, max);
  return static_cast<float>(value);
}

template <typename T, size_t N>
std::vector<T> user_get_array_fixed(Rcpp::List user, const char *name,
                                    const std::vector<T> previous,
                                    const std::array<int, N>& dim,
                                    T min, T max) {
  if (!user.containsElementNamed(name)) {
    if (previous.size() == 0) {
      Rcpp::stop("Expected a value for '%s'", name);
    }
    return previous;
  }

  Rcpp::RObject x = user[name];

  user_check_array_rank<N>(x, name);
  user_check_array_dim<N>(x, name, dim);

  std::vector<T> ret = Rcpp::as<std::vector<T>>(x);
  user_check_array_value(ret, name, min, max);

  return ret;
}

template <typename T, size_t N>
std::vector<T> user_get_array_variable(Rcpp::List user, const char *name,
                                       std::vector<T> previous,
                                       std::array<int, N>& dim,
                                       T min, T max) {
  if (!user.containsElementNamed(name)) {
    if (previous.size() == 0) {
      Rcpp::stop("Expected a value for '%s'", name);
    }
    return previous;
  }

  Rcpp::RObject x = user[name];

  user_check_array_rank<N>(x, name);
  user_set_array_dim<N>(x, name, dim);

  std::vector<T> ret = Rcpp::as<std::vector<T>>(x);
  user_check_array_value(ret, name, min, max);

  return ret;
}

// This is sum with inclusive "from", exclusive "to", following the
// same function in odin
template <typename T>
T odin_sum1(const std::vector<T>& x, size_t from, size_t to) {
  T tot = 0.0;
  for (size_t i = from; i < to; ++i) {
    tot += x[i];
  }
  return tot;
}
template<>
sireinfect::init_t dust_data<sireinfect>(Rcpp::List user) {
  typedef typename sireinfect::real_t real_t;
  sireinfect::init_t internal;
  internal.initial_R = 0;
  internal.I_ini = NA_REAL;
  internal.alpha = 0.10000000000000001;
  internal.beta = 0.20000000000000001;
  internal.gamma = 0.10000000000000001;
  internal.S_ini = 1000;
  internal.alpha = user_get_scalar<real_t>(user, "alpha", internal.alpha, NA_REAL, NA_REAL);
  internal.beta = user_get_scalar<real_t>(user, "beta", internal.beta, NA_REAL, NA_REAL);
  internal.gamma = user_get_scalar<real_t>(user, "gamma", internal.gamma, NA_REAL, NA_REAL);
  internal.I_ini = user_get_scalar<real_t>(user, "I_ini", internal.I_ini, NA_REAL, NA_REAL);
  internal.S_ini = user_get_scalar<real_t>(user, "S_ini", internal.S_ini, NA_REAL, NA_REAL);
  internal.initial_I = internal.I_ini;
  internal.initial_S = internal.S_ini;
  internal.p_IR = 1 - std::exp(- internal.gamma);
  internal.p_RS = 1 - std::exp(- internal.alpha);
  return internal;
}
template <>
Rcpp::RObject dust_info<sireinfect>(const sireinfect::init_t& internal) {
  Rcpp::List ret(3);
  ret[0] = Rcpp::IntegerVector({1});
  ret[1] = Rcpp::IntegerVector({1});
  ret[2] = Rcpp::IntegerVector({1});
  Rcpp::CharacterVector nms = {"S", "I", "R"};
  ret.names() = nms;
  return ret;
}

// [[Rcpp::export(rng = false)]]
SEXP dust_sireinfect_alloc(Rcpp::List r_data, size_t step, size_t n_particles,
                size_t n_threads, size_t seed) {
  return dust_alloc<sireinfect>(r_data, step, n_particles, n_threads,
                                seed);
}

// [[Rcpp::export(rng = false)]]
SEXP dust_sireinfect_run(SEXP ptr, size_t step_end) {
  return dust_run<sireinfect>(ptr, step_end);
}

// [[Rcpp::export(rng = false)]]
SEXP dust_sireinfect_reset(SEXP ptr, Rcpp::List r_data, size_t step) {
  return dust_reset<sireinfect>(ptr, r_data, step);
}

// [[Rcpp::export(rng = false)]]
SEXP dust_sireinfect_state(SEXP ptr, SEXP r_index) {
  return dust_state<sireinfect>(ptr, r_index);
}

// [[Rcpp::export(rng = false)]]
size_t dust_sireinfect_step(SEXP ptr) {
  return dust_step<sireinfect>(ptr);
}

// [[Rcpp::export(rng = false)]]
void dust_sireinfect_reorder(SEXP ptr, Rcpp::IntegerVector r_index) {
  return dust_reorder<sireinfect>(ptr, r_index);
}
