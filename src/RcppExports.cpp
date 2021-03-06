// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <Rcpp.h>

using namespace Rcpp;

// dust_sireinfect_alloc
SEXP dust_sireinfect_alloc(Rcpp::List r_data, size_t step, size_t n_particles, size_t n_threads, size_t n_generators, size_t seed);
RcppExport SEXP _pkg_dust_sireinfect_alloc(SEXP r_dataSEXP, SEXP stepSEXP, SEXP n_particlesSEXP, SEXP n_threadsSEXP, SEXP n_generatorsSEXP, SEXP seedSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::traits::input_parameter< Rcpp::List >::type r_data(r_dataSEXP);
    Rcpp::traits::input_parameter< size_t >::type step(stepSEXP);
    Rcpp::traits::input_parameter< size_t >::type n_particles(n_particlesSEXP);
    Rcpp::traits::input_parameter< size_t >::type n_threads(n_threadsSEXP);
    Rcpp::traits::input_parameter< size_t >::type n_generators(n_generatorsSEXP);
    Rcpp::traits::input_parameter< size_t >::type seed(seedSEXP);
    rcpp_result_gen = Rcpp::wrap(dust_sireinfect_alloc(r_data, step, n_particles, n_threads, n_generators, seed));
    return rcpp_result_gen;
END_RCPP
}
// dust_sireinfect_run
SEXP dust_sireinfect_run(SEXP ptr, size_t step_end);
RcppExport SEXP _pkg_dust_sireinfect_run(SEXP ptrSEXP, SEXP step_endSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::traits::input_parameter< SEXP >::type ptr(ptrSEXP);
    Rcpp::traits::input_parameter< size_t >::type step_end(step_endSEXP);
    rcpp_result_gen = Rcpp::wrap(dust_sireinfect_run(ptr, step_end));
    return rcpp_result_gen;
END_RCPP
}
// dust_sireinfect_reset
SEXP dust_sireinfect_reset(SEXP ptr, Rcpp::List r_data, size_t step);
RcppExport SEXP _pkg_dust_sireinfect_reset(SEXP ptrSEXP, SEXP r_dataSEXP, SEXP stepSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::traits::input_parameter< SEXP >::type ptr(ptrSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type r_data(r_dataSEXP);
    Rcpp::traits::input_parameter< size_t >::type step(stepSEXP);
    rcpp_result_gen = Rcpp::wrap(dust_sireinfect_reset(ptr, r_data, step));
    return rcpp_result_gen;
END_RCPP
}
// dust_sireinfect_state
SEXP dust_sireinfect_state(SEXP ptr, SEXP r_index);
RcppExport SEXP _pkg_dust_sireinfect_state(SEXP ptrSEXP, SEXP r_indexSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::traits::input_parameter< SEXP >::type ptr(ptrSEXP);
    Rcpp::traits::input_parameter< SEXP >::type r_index(r_indexSEXP);
    rcpp_result_gen = Rcpp::wrap(dust_sireinfect_state(ptr, r_index));
    return rcpp_result_gen;
END_RCPP
}
// dust_sireinfect_step
size_t dust_sireinfect_step(SEXP ptr);
RcppExport SEXP _pkg_dust_sireinfect_step(SEXP ptrSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::traits::input_parameter< SEXP >::type ptr(ptrSEXP);
    rcpp_result_gen = Rcpp::wrap(dust_sireinfect_step(ptr));
    return rcpp_result_gen;
END_RCPP
}
// dust_sireinfect_reorder
void dust_sireinfect_reorder(SEXP ptr, Rcpp::IntegerVector r_index);
RcppExport SEXP _pkg_dust_sireinfect_reorder(SEXP ptrSEXP, SEXP r_indexSEXP) {
BEGIN_RCPP
    Rcpp::traits::input_parameter< SEXP >::type ptr(ptrSEXP);
    Rcpp::traits::input_parameter< Rcpp::IntegerVector >::type r_index(r_indexSEXP);
    dust_sireinfect_reorder(ptr, r_index);
    return R_NilValue;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_pkg_dust_sireinfect_alloc", (DL_FUNC) &_pkg_dust_sireinfect_alloc, 6},
    {"_pkg_dust_sireinfect_run", (DL_FUNC) &_pkg_dust_sireinfect_run, 2},
    {"_pkg_dust_sireinfect_reset", (DL_FUNC) &_pkg_dust_sireinfect_reset, 3},
    {"_pkg_dust_sireinfect_state", (DL_FUNC) &_pkg_dust_sireinfect_state, 2},
    {"_pkg_dust_sireinfect_step", (DL_FUNC) &_pkg_dust_sireinfect_step, 1},
    {"_pkg_dust_sireinfect_reorder", (DL_FUNC) &_pkg_dust_sireinfect_reorder, 2},
    {NULL, NULL, 0}
};

RcppExport void R_init_pkg(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
