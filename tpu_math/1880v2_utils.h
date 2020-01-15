#ifndef INC_1880v2_TEST_UTIL_H
#define INC_1880v2_TEST_UTIL_H

#include <libbmutils/bm_debug.h>
#include <libbmruntime/bmruntime_bmkernel.h>
//#include <libbmruntime/bmruntime_bmnet_legacy.h>
//#include "bmruntime_config.h"
#include <bmkernel/bm1880v2/bmkernel_1880v2.h>
//#include <libbmutils/bm_native_ref.h>
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <fstream>
//#include "compression.h"
//#include "bm_vlc_compress.h"
//#include "1880v2_vlc_random_gen_nn_data.h"
//#include <libbmutils/bm_neuron_dump.hpp>
//#include <bmkernel/bm1880v2/1880v2_fp_convert.h>

//#define ENABEL_SIMPLE_BMK1880V2_VLC_TEST
#define ENABEL_GAUSSIANRANDOM_BMK1880V2_VLC_TEST

typedef bmk1880v2_context_t bmk_ctx_t;

typedef bmk1880v2_tensor_lmem_shape_t tl_shape_t;
typedef bmk1880v2_matrix_lmem_shape_t ml_shape_t;
typedef bmk1880v2_tensor_tgmem_shape_t tg_shape_t;
typedef bmk1880v2_matrix_tgmem_shape_t mg_shape_t;

typedef bmk1880v2_tensor_lmem_t tl_t;
typedef bmk1880v2_matrix_lmem_t ml_t;
typedef bmk1880v2_tensor_tgmem_t tg_t;
typedef bmk1880v2_matrix_tgmem_t mg_t;
typedef bmk1880v2_compressed_tensor_tgmem_t compressed_tg_t;
typedef bmk1880v2_compressed_matrix_tgmem_t compressed_mg_t;

typedef bmk1880v2_tensor_tgmem_stride_t tg_stride_t;
typedef bmk1880v2_matrix_tgmem_stride_t mg_stride_t;

typedef struct {
  tg_t tg;
  bmmem_device_t mem;
} tg_wrapper_t;

typedef struct {
  mg_t mg;
  bmmem_device_t mem;
} mg_wrapper_t;

typedef struct {
  compressed_tg_t tg;
  bmmem_device_t mem;
} compressed_tg_wrapper_t;

typedef struct {
  compressed_mg_t mg;
  bmmem_device_t mem;
} compressed_mg_wrapper_t;

typedef enum {
  VLC_CMP_MODE_HW = 0, // <! vlc compress mode - hw, ONLY support bias0/bias1
  VLC_CMP_MODE_COMPILER, // <! vlc compress mode - sw, compiler, it could call bm_vlc_est_weight_bias
  VLC_CMP_MODE_MAX,
} vlc_cmp_mode_e;

typedef struct dim_s {
  int n, c, h, w;
} dim_t;

typedef struct {
  fmt_t src_fmt;
  fmt_t dst_fmt;
} fmt_type;

static inline int dim_size(const dim_t *dim)
{
  return dim->n * dim->c * dim->h * dim->w;
}

static inline u64 tl_shape_size(const tl_shape_t *s)
{
  return (u64)s->n * s->c * s->h * s->w;
}

static inline u64 ml_shape_size(const ml_shape_t *s)
{
  return (u64)s->n * s->col;
}

static inline u64 mg_shape_size(const mg_shape_t *s)
{
  return (u64)s->row * s->col;
}

static inline u64 tg_shape_size(const tg_shape_t *s)
{
  return (u64)s->n * s->c * s->h * s->w;
}

static inline dim_t dim_of_ith_element(int i, dim_t *dim, int transpose)
{
  int channel_offset = i % (dim->h * dim->w);
  int hidx = channel_offset / dim->w;
  int widx = channel_offset % dim->w;
  int channel_index = i / (dim->h * dim->w);
  int nidx = channel_index / dim->c;
  int cidx = channel_index % dim->c;
  if (transpose) {
    nidx = channel_index % dim->n;
    cidx = channel_index / dim->n;
  }
  dim_t r = { nidx, cidx, hidx, widx };
  return r;
}

static inline void * xmalloc(size_t size)
{
  void *p = malloc(size);
  assert(p);
  return p;
}

static inline void test_init(bmctx_t *ctx, bmk_ctx_t **bmk)
{
  int ret = bm_init(0, ctx);
  if (ret != BM_SUCCESS) {
    fprintf(stderr, "bm_init failed, err %d\n", ret);
    exit(-1);
  }

  bmruntime_bmkernel_create(*ctx, (void**)bmk);
}

static inline void test_submit(bmctx_t *ctx)
{
  bmruntime_bmkernel_submit(*ctx);
}

static inline void test_exit(bmctx_t *ctx)
{
  bmruntime_bmkernel_destroy(*ctx);
  bm_exit(*ctx);
}

static inline tl_t * alloc_tl(bmk_ctx_t *bmk, tl_shape_t s, fmt_t f, int align)
{
  tl_t *t = bmk1880v2_lmem_alloc_tensor(bmk, s, f, align);
  t->cmprs_fmt = f;
  assert(t);
  return t;
}

static inline tl_t * alloc_tl_bf16(bmk_ctx_t *bmk, tl_shape_t s, fmt_t f, int align)
{
  tl_t *t = bmk1880v2_lmem_alloc_bf16_tensor(bmk, s, f, align);
  assert(t);
  return t;
}

static inline ml_t * alloc_ml(bmk_ctx_t *bmk, ml_shape_t s, int align)
{
  ml_t *m = bmk1880v2_lmem_alloc_matrix(bmk, s, FMT_I8, align);
  assert(m);
  return m;
}

static inline ml_t * alloc_ml(bmk_ctx_t *bmk, ml_shape_t s, fmt_t f, int align)
{
  ml_t *m = bmk1880v2_lmem_alloc_matrix(bmk, s, f, align);
  assert(m);
  return m;
}

static inline ml_t * alloc_ml_bf16(bmk_ctx_t *bmk, ml_shape_t s, fmt_t f,int align)
{
  ml_t *m = bmk1880v2_lmem_alloc_bf16_matrix(bmk, s, f, align);
  assert(m);
  return m;
}

static inline tg_t * alloc_tg_gmem(bmctx_t *ctx, tg_shape_t s, fmt_t fmt)
{
  bmshape_t bms = BM_TENSOR_INT8(
      (int)s.n,
      (int)s.c,
      (int)s.h,
      (int)s.w);

  tg_wrapper_t *w = new typeof(*w);
  w->mem = bmmem_device_alloc(*ctx, &bms);
  w->tg.base_reg_index = 0;
  w->tg.start_address = bmmem_device_addr(*ctx, w->mem);
  w->tg.fmt = fmt;
  w->tg.shape = s;
  w->tg.stride = bmk1880v2_tensor_tgmem_default_stride(s);

  return &w->tg;
}

static inline tg_t * alloc_tg_bf16_gmem(bmctx_t *ctx, tg_shape_t s, fmt_t fmt)
{
  u32 val = (fmt == FMT_BF16) ? 2 : 1;
  bmshape_t bms = BM_TENSOR_INT8(
      (int)s.n,
      (int)s.c,
      (int)s.h,
      (int)s.w * (int)val);

  tg_wrapper_t *w = new typeof(*w);
  w->mem = bmmem_device_alloc(*ctx, &bms);
  w->tg.base_reg_index = 0;
  w->tg.start_address = bmmem_device_addr(*ctx, w->mem);
  w->tg.fmt = fmt;
  w->tg.shape = s;
  if(fmt == FMT_BF16)
    w->tg.stride = bmk1880v2_bf16_tensor_tgmem_default_stride(s, fmt);
  else
   w->tg.stride = bmk1880v2_tensor_tgmem_default_stride(s);
  return &w->tg;
}

static inline mg_t * alloc_mg_gmem(bmctx_t *ctx, mg_shape_t s)
{
  bmshape_t bms = BM_MATRIX_INT8((int)s.row, (int)s.col);
  mg_wrapper_t *w = new typeof(*w);
  w->mem = bmmem_device_alloc(*ctx, &bms);
  w->mg.base_reg_index = 0;
  w->mg.start_address = bmmem_device_addr(*ctx, w->mem);
  w->mg.shape = s;
  w->mg.stride.row = s.col;

  return &w->mg;
}

static inline compressed_mg_t* alloc_compressed_mg_gmem(bmctx_t *ctx, mg_shape_t s)
{
  bmshape_t bms = BM_MATRIX_INT8((int)s.row, (int)s.col);
  compressed_mg_wrapper_t *w = new typeof(*w);
  w->mem = bmmem_device_alloc(*ctx, &bms);
  w->mg.m.base_reg_index = 0;
  w->mg.m.start_address = bmmem_device_addr(*ctx, w->mem);
  w->mg.m.shape = s;
  w->mg.m.stride.row = s.col;

  return &w->mg;
}

static inline mg_t * alloc_mg_bf16_gmem(bmctx_t *ctx, mg_shape_t s, fmt_t fmt)
{

  u32 val = (fmt == FMT_BF16) ? 2 : 1;
  bmshape_t bms = BM_MATRIX_INT8((int)s.row, (int)s.col * (int)val);
  mg_wrapper_t *w = new typeof(*w);
  w->mem = bmmem_device_alloc(*ctx, &bms);
  w->mg.base_reg_index = 0;
  w->mg.start_address = bmmem_device_addr(*ctx, w->mem);
  w->mg.shape = s;
  w->mg.fmt = fmt;
  w->mg.stride.row = s.col * val;
  //printf("w->mg.stride.row =%x s.col=%x val=%x\n",w->mg.stride.row, s.col, val);
  return &w->mg;
}

// static inline compressed_tg_t * alloc_compressed_tg_gmem(
//     bmctx_t *ctx, tl_shape_t *s, u8 bit_length)
// {
//   u64 size = tl_shape_size(s);
//   u64 header_bytes = 16;
//   u64 map_bytes = compression_map_bytes(size);
//   u64 data_bytes = compression_data_bytes(size, bit_length);
//   u64 total_bytes = header_bytes + map_bytes + data_bytes;
//   compressed_tg_wrapper_t *w = new typeof(*w);
//   w->mem = bmmem_device_alloc_raw(*ctx, total_bytes);
//   w->tg.t.base_reg_index = 0;
//   w->tg.t.start_address = bmmem_device_addr(*ctx, w->mem);
//   w->tg.reserved_size = total_bytes;
//   w->tg.bit_length = bit_length;
//   w->tg.t.shape.n = s->n;
//   w->tg.t.shape.c = s->c;
//   w->tg.t.shape.h = s->h;
//   w->tg.t.shape.w = s->w;
//   w->tg.t.stride = bmk1880v2_tensor_tgmem_default_stride(w->tg.t.shape);
//   return &w->tg;
// }
// <! copy from bmkernel/src/kernel_internal.h
static inline int bitsize_of_fmt(fmt_t fmt)
{
  switch (fmt) {
    case FMT_F32:
    case FMT_I32:
      return 32;
    case FMT_F16:
    case FMT_I16:
    case FMT_U16:
    case FMT_BF16:
      return 16;
    case FMT_I8:
    case FMT_U8:
      return 8;
    case FMT_I4:
      return 4;
    case FMT_I2:
      return 2;
    case FMT_I1:
      return 1;
    default:
      assert(0);
      return -1;
  }
}
// <! end copy from bmkernel/src/kernel_internal.h

static inline int bytesize_of_fmt(fmt_t fmt)
{
  return bitsize_of_fmt(fmt) / 8;
}

// static inline compressed_tg_t * _alloc_vlc_compressed_tg_gmem(
//     bmctx_t *ctx, tl_shape_t *s, fmt_t fmt, CommandInfo* cmd_info)
// {
//   u64 in_size = tl_shape_size(s);

//   u8 data_type = (fmt == FMT_BF16) ? 1 : 0;
//   in_size *= bytesize_of_fmt(fmt);

//   size_t bs_buf_size = get_out_bs_buf_size(in_size, data_type);
//   compressed_tg_wrapper_t *w = new typeof(*w);
//   w->mem = bmmem_device_alloc_raw(*ctx, bs_buf_size);
//   w->tg.t.base_reg_index = 0;
//   w->tg.t.start_address = bmmem_device_addr(*ctx, w->mem);
//   w->tg.reserved_size = bs_buf_size;
//   w->tg.t.fmt = fmt;

//   if (cmd_info) {
//     w->tg.bias0 = cmd_info->bias0;
//     w->tg.bias1 = cmd_info->bias1;
//     w->tg.zero_guard_en = cmd_info->zero_guard_en;
//   }
//   else {
//     if (fmt == FMT_BF16) {
//       w->tg.bias0 = 127;
//     }
//     else if (fmt == FMT_I8 || fmt == FMT_U8) {
//       w->tg.bias0 = 0;
//     }
//     else {
//       printf("only accept fmt for FMT_BF16/FMT_I8/FMT_U8/, your format is %d\n", fmt);
//       assert(0);
//     }

//     w->tg.bias1 = 0;
//     // <! TODO: need to analyze data contain 0
//     w->tg.zero_guard_en = 0;
//   }
//   w->tg.t.shape.n = s->n;
//   w->tg.t.shape.c = s->c;
//   w->tg.t.shape.h = s->h;
//   w->tg.t.shape.w = s->w;

//   if(fmt == FMT_BF16)
//     w->tg.t.stride = bmk1880v2_bf16_tensor_tgmem_default_stride(w->tg.t.shape, fmt);
//   else
//     w->tg.t.stride = bmk1880v2_tensor_tgmem_default_stride(w->tg.t.shape);

//   return &w->tg;
// }

// static inline compressed_tg_t * alloc_vlc_compressed_tg_gmem(
//     bmctx_t *ctx, tl_shape_t *s, fmt_t fmt)
// {
//   return _alloc_vlc_compressed_tg_gmem(ctx, s, fmt, NULL);
// }

// /**
//  * \shape_size shape size
//  * \signedness 0 means ungiend 1 means signed
//  * \data_type 0 means 8bit 1 means bf16
//  */
// static inline void vlc_init_testdata(u16 *src_data, u64 shape_size, bool signedness, bool data_type) {
// #ifdef ENABEL_GAUSSIANRANDOM_BMK1880V2_VLC_TEST
//   float zero_ratio = 0;
//   assert(signedness == 0); //<! bf16 only set to 0
//   assert(data_type == 1); //<! bf16 only set to 1
//   random_gen_nn_data((u8* )src_data, shape_size, signedness, data_type, zero_ratio);
// #else /* ! ifdef ENABEL_GAUSSIANRANDOM_BMK1880V2_VLC_TEST */
//   printf ("randome signedness %d data_type %d\n", signedness, data_type);
//   memset(src_data, 0x00, shape_size * sizeof(u16));
//   for (u64 i = 0; i < shape_size; i++)
//     src_data[i] = 200 + i;

//   u64 zero_range = 20; //<! friendly enhance compress ratio
//   if (shape_size > zero_range) {
//     for (u64 i = 0; i < shape_size - zero_range; i++) {
//       src_data[i] = 0;
//     }
//   }
// #endif /* ifdef ENABEL_GAUSSIANRANDOM_BMK1880V2_VLC_TEST */
// }

// static inline void vlc_init_testdata(u8 *src_data, u64 shape_size, bool signedness, bool data_type) {
//   memset(src_data, 0x00, shape_size);
// #ifdef ENABEL_GAUSSIANRANDOM_BMK1880V2_VLC_TEST
//   float zero_ratio = 0;
//   assert(data_type == 0); //<! bf16 only set to 1
//   random_gen_nn_data(src_data, shape_size, signedness, data_type, zero_ratio);
// #else /* ! ifdef ENABEL_GAUSSIANRANDOM_BMK1880V2_VLC_TEST */
//   for (u64 i = 0; i < shape_size; i++)
//     src_data[i] = 200 + i;

//   u64 zero_range = 20; //<! friendly enhance compress ratio
//   if (shape_size > zero_range) {
//     for (u64 i = 0; i < shape_size - zero_range; i++) {
//       src_data[i] = 0;
//     }
//   }
// #endif /* ifdef ENABEL_GAUSSIANRANDOM_BMK1880V2_VLC_TEST */
// }

// static inline compressed_mg_t * alloc_vlc_compressed_mg_gmem(
//     bmctx_t *ctx, mg_shape_t s, fmt_t fmt, CommandInfo* cmd_info)
// {
//   u64 in_size = mg_shape_size(&s);
//   u8 data_type = (fmt == FMT_BF16) ? 1 : 0;
//   in_size *= bytesize_of_fmt(fmt);

//   size_t bs_buf_size = get_out_bs_buf_size(in_size, data_type);

//   compressed_mg_wrapper_t *w = new typeof(*w);

//   w->mem = bmmem_device_alloc_raw(*ctx, bs_buf_size);
//   w->mg.m.shape = s;
//   w->mg.m.stride.row = s.col * bytesize_of_fmt(fmt);
//   w->mg.m.base_reg_index = 0;
//   w->mg.m.fmt = fmt;
//   w->mg.m.start_address = bmmem_device_addr(*ctx, w->mem);

//   if (cmd_info) {
//     w->mg.bias0 = cmd_info->bias0;
//     w->mg.bias1 = cmd_info->bias1;
//     w->mg.zero_guard_en = cmd_info->zero_guard_en;
//   }
//   else {
//     w->mg.bias0 = 0;

//     if (fmt == FMT_BF16) {
//       w->mg.bias0 = 127;
//     }
//     else if (fmt == FMT_I8 || fmt == FMT_U8) {
//       w->mg.bias0 = 0;
//     }
//     else {
//       printf("only accept fmt for FMT_BF16/FMT_I8/FMT_U8/, your format is %d\n", fmt);
//       assert(0);
//     }

//     w->mg.bias1 = 0;
//     // <! FIXME: need to analyze data contain 0
//     w->mg.zero_guard_en = 0;
//   }

//   return &w->mg;
// }

// /**
//  * \cmd_info_est_in that manual set compress parameters, the possible input as below
//     1. NULL, it could call \bm_vlc_est_weight_bias
//     2. not NULL that directly send to \bm_vlc_enc_int8
//  * \cmd_info_est_out output est result, the passble value as following
//     1. \cmd_info_est_out = \cmd_info_est_in once cmd_info_est_in != NULL
//     2. \cmd_info_est_out = est result once cmd_info_est_in == NULL
//     3. NULL if you dont care
//  */
// static inline u8 *vlc_compress (
//     u8 *src_data, u64 size, int is_signed, int data_type, size_t* bs_size, const CommandInfo* cmd_info_est_in, CommandInfo* cmd_info_est_out)
// {
//   CommandInfo cmd_info;
//   size_t bs_buf_size = get_out_bs_buf_size(size, data_type);

//   u8 *bsbuf = new u8[bs_buf_size];
//   memset(&cmd_info, 0x00, sizeof(CommandInfo));

//   /* generate comparess data (bsbuf)*/
//   if (cmd_info_est_in) {
//     memcpy(&cmd_info, cmd_info_est_in, sizeof(CommandInfo));
//   }
//   else {
//     bm_vlc_est_weight_bias(src_data, size, (bool)is_signed, (bool)data_type, &cmd_info);
//   }

//   if (cmd_info_est_out) {
//     memcpy(cmd_info_est_out, &cmd_info, sizeof(CommandInfo));
//   }

//   if (data_type) {
//     bm_vlc_enc_bf16((u16*)src_data, size, bsbuf, bs_size, &cmd_info);
//   }
//   else {
//     bm_vlc_enc_int8(src_data, size, bsbuf, bs_size, &cmd_info);
//   }

//   return bsbuf;
// }

// static inline int get_vlc_compressed_meta(
//     u8 *src_data, u64 in_size, fmt_t fmt, size_t* bs_size, CommandInfo* cmd_info)
// {
//   int is_signed = (fmt == FMT_I8);
//   int data_type = (fmt == FMT_BF16) ? 1 : 0;
//   //bm_vlc_est_weight_bias(src_data, in_size, (bool)is_signed, (bool)data_type, cmd_info);

//   u8 *ref_data = vlc_compress(src_data, in_size, is_signed, data_type, bs_size, cmd_info, NULL);
//   delete[] ref_data;
//   return 0;
// }

static inline void free_tl(bmk_ctx_t *bmk, const tl_t *t)
{
  return bmk1880v2_lmem_free_tensor(bmk, t);
}

static inline void free_ml(bmk_ctx_t *bmk, const ml_t *m)
{
  return bmk1880v2_lmem_free_matrix(bmk, m);
}

static inline void free_tg_gmem(bmctx_t *ctx, const tg_t *tg)
{
  tg_wrapper_t *w = (typeof(w))tg;
  bmmem_device_free(*ctx, w->mem);
  delete w;
}

static inline void free_mg_gmem(bmctx_t *ctx, const mg_t *mg)
{
  mg_wrapper_t *w = (typeof(w))mg;
  bmmem_device_free(*ctx, w->mem);
  delete w;
}

static inline void free_compressed_tg_gmem(
    bmctx_t *ctx, const compressed_tg_t *t)
{
  compressed_tg_wrapper_t *w = (typeof(w))t;
  bmmem_device_free(*ctx, w->mem);
  delete w;
}

static inline void free_compressed_mg_gmem(
    bmctx_t *ctx, const compressed_mg_t *t)
{
  compressed_mg_wrapper_t *w = (typeof(w))t;
  bmmem_device_free(*ctx, w->mem);
  delete w;
}

static inline u8 * get_tg_gmem(bmctx_t *ctx, const tg_t *tg)
{
  tg_shape_t s = tg->shape;
  u32 size = s.n * s.c * s.h * s.w;
  u8 *data = new u8[size];

  tg_wrapper_t *w = (typeof(w))tg;
  int ret = bm_memcpy_d2s(*ctx, data, w->mem);
  assert(ret == BM_SUCCESS);

  return data;
}

static inline u8 * get_tg_bf16_gmem(bmctx_t *ctx, const tg_t *tg)
{
  tg_shape_t s = tg->shape;
  u32 size = s.n * s.c * s.h * s.w * (tg->fmt == FMT_BF16 ? 2 : 1);
  u8 *data = new u8[size];
  tg_wrapper_t *w = (typeof(w))tg;
  int ret = bm_memcpy_d2s(*ctx, data, w->mem);
  assert(ret == BM_SUCCESS);

  return data;
}

static inline u8 * get_mg_gmem(bmctx_t *ctx, const mg_t *mg)
{
  mg_shape_t s = mg->shape;
  u32 size = s.row * s.col;
  u8 *data = new u8[size];

  mg_wrapper_t *w = (typeof(w))mg;
  int ret = bm_memcpy_d2s(*ctx, data, w->mem);
  assert(ret == BM_SUCCESS);

  return data;
}

static inline u8 * get_compressed_mg_gmem(bmctx_t *ctx, const compressed_mg_t *mg, size_t bs_size)
{
  //mg_shape_t s = mg->m.shape;
  //u32 size = s.row * s.col;
  u8 *data = new u8[bs_size];

  compressed_mg_wrapper_t *w = (typeof(w))mg;
  int ret = bm_memcpy_d2s(*ctx, data, w->mem);
  assert(ret == BM_SUCCESS);

  return data;
}

static inline u8 * get_mg_bf16_gmem(bmctx_t *ctx, const mg_t *mg)
{
  mg_shape_t s = mg->shape;
  u32 size = s.row * s.col * (mg->fmt == FMT_BF16 ? 2 : 1);
  u8 *data = new u8[size];

  mg_wrapper_t *w = (typeof(w))mg;
  int ret = bm_memcpy_d2s(*ctx, data, w->mem);
  assert(ret == BM_SUCCESS);

  return data;
}

static inline u8 * get_compressed_tg_gmem(
    bmctx_t *ctx, const compressed_tg_t *t)
{
  compressed_tg_wrapper_t *w = (typeof(w))t;

  u8 *data = new u8[t->reserved_size];
  int ret = bm_memcpy_d2s(*ctx, data, w->mem);
  assert(ret == BM_SUCCESS);

  return data;
}

static inline u8 * get_bytes_gmem(bmctx_t *ctx, u64 addr, u64 size)
{
  bmmem_device_t mem = bmmem_device_prealloc_raw(*ctx, NULL, addr, size);

  u8 *data = new u8[size];
  int ret = bm_memcpy_d2s(*ctx, data, mem);
  assert(ret == BM_SUCCESS);

  return data;
}

static inline u8 * get_bytes_bf16_gmem(bmctx_t *ctx, u64 addr, u64 size, fmt_t data_format)
{
  bmmem_device_t mem = bmmem_device_prealloc_raw(*ctx, NULL, addr, size);
  u32 val = (data_format == FMT_BF16) ? 2 : 1;
  u8 *data = new u8[size * val];
  int ret = bm_memcpy_d2s(*ctx, data, mem);
  assert(ret == BM_SUCCESS);

  return data;
}



static inline void put_tg_gmem(bmctx_t *ctx, const tg_t *tg, u8 data[])
{
  tg_wrapper_t *w = (typeof(w))tg;
  int ret = bm_memcpy_s2d(*ctx, w->mem, data);
  assert(ret == BM_SUCCESS);
}

static inline void put_tg_bf16_gmem(bmctx_t *ctx, const tg_t *tg, u8 data[])
{
  tg_wrapper_t *w = (typeof(w))tg;
  int ret = bm_memcpy_s2d(*ctx, w->mem, data);
  assert(ret == BM_SUCCESS);
}

static inline void put_mg_gmem(bmctx_t *ctx, const mg_t *mg, u8 data[])
{
  mg_wrapper_t *w = (typeof(w))mg;
  int ret = bm_memcpy_s2d(*ctx, w->mem, data);
  assert(ret == BM_SUCCESS);
}

static inline void put_mg_bf16_gmem(bmctx_t *ctx, const mg_t *mg, u8 data[])
{
  mg_wrapper_t *w = (typeof(w))mg;
  int ret = bm_memcpy_s2d(*ctx, w->mem, data);
  assert(ret == BM_SUCCESS);
}

static inline void put_bytes_gmem(bmctx_t *ctx, u64 addr, u64 size, u8 data[])
{
  bmmem_device_t mem = bmmem_device_prealloc_raw(*ctx, NULL, addr, size);

  int ret = bm_memcpy_s2d(*ctx, mem, data);
  assert(ret == BM_SUCCESS);
}

static inline void put_compressed_tg_gmem(
    bmctx_t *ctx, const compressed_tg_t *t, u8 buf[], u64 size)
{
  assert(size <= t->reserved_size);

  compressed_tg_wrapper_t *w = (typeof(w))t;
  u64 addr = bmmem_device_addr(*ctx, w->mem);

  put_bytes_gmem(ctx, addr, size, buf);
}

static inline void put_compressed_mg_gmem(
    bmctx_t *ctx, const compressed_mg_t *t, u8 buf[], u64 size)
{
  compressed_mg_wrapper_t *w = (typeof(w))t;
  u64 addr = bmmem_device_addr(*ctx, w->mem);

  put_bytes_gmem(ctx, addr, size, buf);
}

static inline void put_tensor_g2l(
    bmctx_t *ctx, bmk_ctx_t *bmk, const tl_t *tl, u8 data[])
{
  tg_shape_t s;
  s.n = tl->shape.n;
  s.c = tl->shape.c;
  s.h = tl->shape.h;
  s.w = tl->shape.w;
  tg_t *tg = alloc_tg_gmem(ctx, s, FMT_I8);

  bmk1880v2_tdma_tg2l_tensor_copy_param_t p;
  p.src = tg;
  p.dst = tl;

  put_tg_gmem(ctx, tg, data);
  bmk1880v2_tdma_g2l_tensor_copy(bmk, &p);
  test_submit(ctx);

  free_tg_gmem(ctx, tg);
}

/**
 * prepard mean you alloc address but not submit it
 * once submit it could re-assign from head
 */
static inline tg_t* prepare_put_bf16_tensor_g2l(
    bmctx_t *ctx, bmk_ctx_t *bmk, const tl_t *tl, u16 data[], fmt_t tg_data_format,
bmk1880v2_tdma_tg2l_tensor_copy_param_t* p)
{
  tg_shape_t s;
  s.n = tl->shape.n;
  s.c = tl->shape.c;
  s.h = tl->shape.h;
  s.w = tl->shape.w;
  tg_t *tg = alloc_tg_bf16_gmem(ctx, s, tg_data_format);

  p->src = tg;
  p->dst = tl;

  assert(bmk);

  put_tg_bf16_gmem(ctx, tg, (u8 *)data);
  return tg;
}

/**
 * issue prepared one
 */
static inline void launch_put_bf16_tensor_g2l(bmctx_t *ctx, bmk_ctx_t *bmk,
const tg_t *tg, bmk1880v2_tdma_tg2l_tensor_copy_param_t* p) {
  bmk1880v2_tdma_g2l_bf16_tensor_copy(bmk, p);
  test_submit(ctx);
  free_tg_gmem(ctx, tg);
}

static inline void put_bf16_tensor_g2l(
    bmctx_t *ctx, bmk_ctx_t *bmk, const tl_t *tl, u16 data[], fmt_t tg_data_format)
{
  tg_shape_t s;
  s.n = tl->shape.n;
  s.c = tl->shape.c;
  s.h = tl->shape.h;
  s.w = tl->shape.w;
  tg_t *tg = alloc_tg_bf16_gmem(ctx, s, tg_data_format);

  bmk1880v2_tdma_tg2l_tensor_copy_param_t p;
  p.src = tg;
  p.dst = tl;

  put_tg_bf16_gmem(ctx, tg, (u8 *)data);
  bmk1880v2_tdma_g2l_bf16_tensor_copy(bmk, &p);
  test_submit(ctx);
  free_tg_gmem(ctx, tg);
}

static inline void put_matrix_g2l(
    bmctx_t *ctx, bmk_ctx_t *bmk, const ml_t *ml, u8 data[])
{
  mg_shape_t s;
  s.row = ml->shape.n;
  s.col = ml->shape.col;
  mg_t *mg = alloc_mg_gmem(ctx, s);

  bmk1880v2_tdma_tg2l_matrix_copy_param_t p;
  p.src = mg;
  p.dst = ml;

  put_mg_gmem(ctx, mg, data);
  bmk1880v2_tdma_g2l_matrix_copy(bmk, &p);
  test_submit(ctx);

  free_mg_gmem(ctx, mg);
}


static inline void put_bf16_matrix_g2l(
    bmctx_t *ctx, bmk_ctx_t *bmk, const ml_t *ml, u8 data[], fmt_t mg_data_format)
{
  mg_shape_t s;
  s.row = ml->shape.n;
  s.col = ml->shape.col;
  mg_t *mg = alloc_mg_bf16_gmem(ctx, s, mg_data_format);

  bmk1880v2_tdma_tg2l_matrix_copy_param_t p;
  p.src = mg;
  p.dst = ml;

  put_mg_bf16_gmem(ctx, mg, data);
  bmk1880v2_tdma_g2l_bf16_matrix_copy(bmk, &p);
  test_submit(ctx);

  free_mg_gmem(ctx, mg);
}

static inline void put_bytes_g2l(
    bmctx_t *ctx, bmk_ctx_t *bmk, u32 lmem_addr, u64 size, u8 data[])
{
  bmmem_device_t mem = bmmem_device_alloc_raw(*ctx, size);
  u64 gmem_addr = bmmem_device_addr(*ctx, mem);

  bmk1880v2_tdma_tg2l_general_copy_param_t p;
  p.src_base_reg_index = 0;
  p.src_address = gmem_addr;
  p.dst_address = lmem_addr;
  p.bytes = size;

  put_bytes_gmem(ctx, gmem_addr, size, data);
  bmk1880v2_tdma_g2l_general_copy(bmk, &p);
  test_submit(ctx);

  bmmem_device_free(*ctx, mem);
}

static inline u8 * get_tensor_l2g(bmctx_t *ctx, bmk_ctx_t *bmk, const tl_t *tl)
{
  tg_shape_t s;
  s.n = tl->shape.n;
  s.c = tl->shape.h;
  s.h = tl->shape.w;
  s.w = tl->shape.c;
  tg_t *tg = alloc_tg_gmem(ctx, s, FMT_I8);

  bmk1880v2_tdma_l2tg_tensor_copy_param_t p;
  p.src = tl;
  p.dst = tg;

  bmk1880v2_tdma_l2g_tensor_copy(bmk, &p);
  test_submit(ctx);
  u8 *data = get_tg_gmem(ctx, tg);

  free_tg_gmem(ctx, tg);
  return data;
}

static inline u8 * get_bf16_tensor_l2g(bmctx_t *ctx, bmk_ctx_t *bmk, const tl_t *tl, fmt_t tg_data_format)
{
  tg_shape_t s;
  s.n = tl->shape.n;
  s.c = tl->shape.h;
  s.h = tl->shape.w;
  s.w = tl->shape.c;

  tg_t *tg = alloc_tg_bf16_gmem(ctx, s, tg_data_format); // alloc tg to bf16 or int8 mode

  bmk1880v2_tdma_l2tg_tensor_copy_param_t p;
  p.src = tl;
  p.dst = tg;

  bmk1880v2_tdma_l2g_bf16_tensor_copy(bmk, &p);
  test_submit(ctx);
  u8 *data = get_tg_bf16_gmem(ctx, tg);

  free_tg_gmem(ctx, tg);
  return data;
}


static inline u8 * get_matrix_l2g(bmctx_t *ctx, bmk_ctx_t *bmk, const ml_t *ml)
{
  mg_shape_t s;
  s.row = ml->shape.n;
  s.col = ml->shape.col;
  mg_t *mg = alloc_mg_gmem(ctx, s);

  bmk1880v2_tdma_l2tg_matrix_copy_param_t p;
  p.src = ml;
  p.dst = mg;

  bmk1880v2_tdma_l2g_matrix_copy(bmk, &p);
  test_submit(ctx);
  u8 *data = get_mg_gmem(ctx, mg);

  free_mg_gmem(ctx, mg);
  return data;
}

static inline u8 * get_bf16_matrix_l2g(bmctx_t *ctx, bmk_ctx_t *bmk, const ml_t *ml, fmt_t mg_data_format)
{
  mg_shape_t s;
  s.row = ml->shape.n;
  s.col = ml->shape.col;
  mg_t *mg = alloc_mg_bf16_gmem(ctx, s, mg_data_format);

  bmk1880v2_tdma_l2tg_matrix_copy_param_t p;
  p.src = ml;
  p.dst = mg;

  bmk1880v2_tdma_l2g_bf16_matrix_copy(bmk, &p);
  test_submit(ctx);
  u8 *data = get_mg_bf16_gmem(ctx, mg);

  free_mg_gmem(ctx, mg);
  return data;
}

static inline u8 * get_bytes_l2g(
    bmctx_t *ctx, bmk_ctx_t *bmk, u32 lmem_addr, u64 size)
{
  bmmem_device_t mem = bmmem_device_alloc_raw(*ctx, size);
  u64 gmem_addr = bmmem_device_addr(*ctx, mem);

  bmk1880v2_tdma_l2tg_general_copy_param_t p;
  p.src_address = lmem_addr;
  p.dst_base_reg_index = 0;
  p.dst_address = gmem_addr;
  p.bytes = size;

  bmk1880v2_tdma_l2g_general_copy(bmk, &p);
  test_submit(ctx);
  u8 *data = get_bytes_gmem(ctx, gmem_addr, size);

  bmmem_device_free(*ctx, mem);
  return data;
}

/*
 * tensor dump utility
 * detail = 1, dump all tensor and indicate N and C number
 * detail = 0, only dump 3 byte closing to begin and end point.
 */
static inline void dump_tensor(u8 src[], u32 n, u32 c, u32 h, u32 w, u8 detail)
{
  if (detail) {
    for (u32 ni = 0; ni < n; ni++) {
      for (u32 ci = 0; ci < c; ci++) {
        for (u32 hi = 0; hi < h; hi++) {
          for (u32 wi = 0; wi < w; wi++) {
            u32 i = ni * c * h * w + ci * h * w + hi * w + wi;
            printf("%4d ", src[i]);

            if (hi == 0 && wi == w-1)
              printf("| <= C: %d ", ci);

            if (ci == 0 && hi == 0 && wi == w-1)
              printf("@ <= N: %d ", ni);
          }
          printf("\n");
        }
      }
    }
  } else {
    u64 end = (n-1) * c * h * w + (c-1) * h * w + (h-1) * w + (w-1);
    printf("[");
    printf("%4d", src[0]);
    printf("%4d", src[1]);
    printf("%4d", src[2]);
    printf(" ... ");
    printf("%4d", src[end - 2]);
    printf("%4d", src[end - 1]);
    printf("%4d", src[end]);
    printf(" ]\n");
  }
}

static inline void saturate_to_int8(s32 *buf, u64 size, int res_sign)
{
  s32 max, min;
  if (res_sign) {
    max = 127;
    min = -128;
  } else {
    max = 255;
    min = 0;
  }

  for (u64 i = 0; i < size; i++) {
    if (buf[i] > max)
      buf[i] = max;
    else if (buf[i] < min)
      buf[i] = min;
  }
}

static inline void saturate_to_int16(s32 *buf, u64 size, int res_sign)
{
  s32 max, min;
  if (res_sign) {
    max = 32767;
    min = -32768;
  } else {
    max = 65535;
    min = 0;
  }

  for (u64 i = 0; i < size; i++) {
    if (buf[i] > max)
      buf[i] = max;
    else if (buf[i] < min)
      buf[i] = min;
  }
}

static inline void arith_right_shift(
    s32 *buf, u64 size, int shift_bits, int round_up)
{
  if (shift_bits == 0)
    return;

  for (u64 i = 0; i < size; i++) {
    buf[i] >>= shift_bits - 1;
    if (round_up)
      buf[i] += 1;
    buf[i] >>= 1;
  }
}

static inline void logic_right_shift(
    s32 *buf, u64 size, int shift_bits, int round_up)
{
  if (shift_bits == 0)
    return;

  for (u64 i = 0; i < size; i++) {
    buf[i] = (u32)buf[i] >> (shift_bits - 1);
    if (round_up)
      buf[i] += 1;
    buf[i] = (u32)buf[i] >> 1;
  }
}

static inline void relu(s32 *buf, u64 size)
{
  for (u64 i = 0; i < size; i++)
    if (buf[i] < 0)
      buf[i] = 0;
}

#endif /* INC_1880v2_TEST_UTIL_H */
