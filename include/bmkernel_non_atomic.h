#pragma once
// TODO: Move into bmkernel.

#include <bmkernel/bm1880v2/bmkernel_1880v2.h>
#include <libbmruntime/bmruntime_bmkernel.h>

struct bmk1880v2_tiu_non_atomic_sqrt_param_t {
  bmk1880v2_tensor_lmem_t *a;
  bmk1880v2_tensor_lmem_t *res;
  bmk1880v2_tensor_lmem_t *buf;
  bmk1880v2_tensor_lmem_t *sqrt_table_answer;
  bmk1880v2_tensor_lmem_t *sqrt_table_answer_mantissa;
};

struct bmk1880v2_tiu_non_atomic_atan2_param_t {
  bmk1880v2_tensor_lmem_t *a;
  bmk1880v2_tensor_lmem_t *b;
  bmk1880v2_tensor_lmem_t *res;
  bmk1880v2_tensor_lmem_t *buf1;
  bmk1880v2_tensor_lmem_t *buf2;
  bmk1880v2_tensor_lmem_t *buf3;
  bmk1880v2_tensor_lmem_t *y0;
  bmk1880v2_tensor_lmem_t *slope;
  bmk1880v2_tensor_lmem_t *invert;
  bmk1880v2_tensor_lmem_t *pos_neg;
  bmk1880v2_tensor_lmem_t *reciprocal_table_answer;
  bmk1880v2_tensor_lmem_t *reciprocal_table_answer_mantissa;
  bmk1880v2_tensor_lmem_t *sqrt_table_answer;
  bmk1880v2_tensor_lmem_t *sqrt_table_answer_mantissa;
  fmt_t fmt;
};