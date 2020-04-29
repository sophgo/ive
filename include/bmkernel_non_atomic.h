#pragma once
// TODO: Move into bmkernel.

#include <bmkernel/bm1880v2/bmkernel_1880v2.h>
#include <bmruntime.h>

typedef struct bmk1880v2_tiu_non_atomic_sqrt_param {
  bmk1880v2_tensor_lmem_t *a;
  bmk1880v2_tensor_lmem_t *res;
  bmk1880v2_tensor_lmem_t *buf;
  bmk1880v2_tensor_lmem_t *sqrt_table_answer;
  bmk1880v2_tensor_lmem_t *sqrt_table_answer_mantissa;
} bmk1880v2_tiu_non_atomic_sqrt_param_t;

typedef struct bmk1880v2_tiu_non_atomic_atan2_param {
  bmk1880v2_tensor_lmem_t *a;
  bmk1880v2_tensor_lmem_t *b;
  bmk1880v2_tensor_lmem_t *res;
  bmk1880v2_tensor_lmem_t *buf1;
  bmk1880v2_tensor_lmem_t *buf2;
  bmk1880v2_tensor_lmem_t *buf3;
  bmk1880v2_tensor_lmem_t *buf4;
  bmk1880v2_tensor_lmem_t *buf5;
  bmk1880v2_tensor_lmem_t *buf6;
  bmk1880v2_tensor_lmem_t *y0;
  bmk1880v2_tensor_lmem_t *slope;
  bmk1880v2_tensor_lmem_t *invert;
  bmk1880v2_tensor_lmem_t *pos_neg_table;
  bmk1880v2_tensor_lmem_t *reciprocal_table_answer;
  bmk1880v2_tensor_lmem_t *reciprocal_table_answer_mantissa;
  bmk1880v2_tensor_lmem_t *sqrt_table_answer;
  bmk1880v2_tensor_lmem_t *sqrt_table_answer_mantissa;
  bmk1880v2_tensor_lmem_t *idx_0_table;
  fmt_t fmt;
  bool output_degree;
} bmk1880v2_tiu_non_atomic_atan2_param_t;

typedef struct bmk1880v2_tiu_non_atomic_mask_param {
  bmk1880v2_tensor_lmem_t *ifmap;
  bmk1880v2_tensor_lmem_t *ofmap;
  bmk1880v2_tensor_lmem_t *buf;
  bmk1880v2_tensor_lmem_t *buf2;
  bmk1880v2_tensor_lmem_t *buf3;
  bmk1880v2_tensor_lmem_t *pos_neg_table;
  bmk1880v2_tensor_lmem_t *idx_0_table;
  fmt_t fmt;
} bmk1880v2_tiu_non_atomic_mask_param_t;