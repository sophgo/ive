#pragma once
#include "debug.hpp"
#include "tpu_data.hpp"

#include <bmkernel/bm1880v2/1880v2_fp_convert.h>
#include <bmruntime.h>

#include <assert.h>
#include <limits.h>
#include <string.h>
#include <iostream>
#include <neon_utils.hpp>
#include <vector>

#define MULTIPLIER_ONLY_PACKED_DATA_SIZE 5

inline int createHandle(bmctx_t *ctx, cvk_context_t **cvk_ctx) {
  int ret = bm_init(0, ctx);
  if (ret != BM_SUCCESS) {
    fprintf(stderr, "cvi_init failed, err %d\n", ret);
    return CVI_FAILURE;
  }
  cvk_reg_info_t req_info;
  strncpy(req_info.chip_ver_str, "cv1880v2", sizeof(req_info.chip_ver_str) - 1);
  req_info.cmdbuf_size = 0x10000000;
  req_info.cmdbuf = static_cast<uint8_t *>(malloc(req_info.cmdbuf_size));
  if (!req_info.cmdbuf) {
    printf("cmdbuf init failed. Insufficient memory %x.\n", req_info.cmdbuf_size);
    return CVI_FAILURE;
  }
  // req_info.cmdbuf = bmk_info_.cmdbuf;
  *cvk_ctx = cvikernel_register(&req_info);
  if (!*cvk_ctx) {
    printf("cmdbuf register failed.\n");
    return CVI_FAILURE;
  }
  return CVI_SUCCESS;
}

inline void destroyHandle(bmctx_t *ctx, cvk_context_t *cvk_ctx) {
  if (cvk_ctx) {
    cvk_ctx->ops->cleanup(cvk_ctx);
  }
  bm_exit(*ctx);
}

inline void submitCmdbuf(bmctx_t *ctx, cvk_context_t *cvk_ctx, const std::string &cmdbuf_subfix,
                         bool write_cmdbuf = false) {
  uint32_t len;
  uint8_t *buf = cvk_ctx->ops->acquire_cmdbuf(cvk_ctx, &len);
  if (write_cmdbuf) {
    printf("Cmdbuf length %u\n", len);
    FILE *pFile;
    std::string name = cmdbuf_subfix == "" ? "cmdbuf.bin" : "cmdbuf_" + cmdbuf_subfix + ".bin";
    pFile = fopen(name.c_str(), "wb");
    fwrite(buf, sizeof(char), len, pFile);
    fclose(pFile);
  }
  uint16_t seq_no;
  bm_send_cmdbuf(*ctx, buf, (size_t)len, &seq_no);
  cvk_ctx->ops->reset(cvk_ctx);
}

inline void genTableU8(const cvk_tl_shape_t &table_shape, const uint8_t *table_data,
                       uint8_t *tg_table) {
  int table_hw = table_shape.h * table_shape.w;

  // duplicate channel #0 to #31
  // TODO: tensor copy
  for (uint64_t i = 0; i < table_shape.c; i++) {
    memcpy(&tg_table[table_hw * i], table_data, sizeof(uint8_t) * table_hw);
  }
}

inline void genTableBF16(const cvk_tl_shape_t &table_shape, const float min_value,
                         const float max_value, uint16_t *table_pos_neg) {
  uint32_t half = table_shape.h * table_shape.w / 2;
  int table_hw = table_shape.h * table_shape.w;

  // data >= 0
  for (uint32_t i = 0; i < half; i++) {
    table_pos_neg[i] = convert_fp32_bf16(max_value);
  }

  // data < 0
  for (uint32_t i = half; i < half * 2; i++) {
    table_pos_neg[i] = convert_fp32_bf16(min_value);
  }

  // duplicate channel #1 to #31
  // TODO: tensor copy
  for (uint64_t i = 1; i < table_shape.c; i++) {
    memcpy(&table_pos_neg[table_hw * i], &table_pos_neg[0], sizeof(uint16_t) * table_hw);
  }
}

inline bool tgTLShapeCompare(cvk_tl_shape_t &tl_shape, cvk_tg_shape_t &tg_shape) {
  if (tg_shape.n == tl_shape.n && tg_shape.c == tl_shape.c && tg_shape.h == tl_shape.h &&
      tg_shape.w == tl_shape.w) {
    return true;
  }
  return false;
}

inline void cviImgFlush2TL(bmctx_t *ctx, cvk_context_t *cvk_ctx, CviImg &img, cvk_tl_t *lmem) {
  img.Flush(ctx);
  cvk_tdma_g2l_tensor_copy_param_t p;
  p.src = &img.m_tg;
  p.dst = lmem;
  cvk_ctx->ops->tdma_g2l_bf16_tensor_copy(cvk_ctx, &p);
}

inline void cviImg2TL(bmctx_t *ctx, cvk_context_t *cvk_ctx, const CviImg &img, cvk_tl_t *lmem) {
  cvk_tdma_g2l_tensor_copy_param_t p;
  p.src = &img.m_tg;
  p.dst = lmem;
  cvk_ctx->ops->tdma_g2l_bf16_tensor_copy(cvk_ctx, &p);
}

inline void constantFillTL(bmctx_t *ctx, cvk_context_t *cvk_ctx, const uint16_t value,
                           cvk_tl_t *lmem) {
  cvk_tdma_g2l_tensor_fill_constant_param_t p_fill;
  p_fill.constant = value;
  p_fill.dst = lmem;

  cvk_ctx->ops->tdma_g2l_bf16_tensor_fill_constant(cvk_ctx, &p_fill);
}

// FIXME: O0 may crash. Reason unknown.
inline void bf16LookupTable(cvk_context_t *cvk_ctx, const cvm_tiu_mask_param_t *mask) {
  cvk_tdma_l2l_tensor_copy_param_t p10;
  cvk_tl_t lmem = *mask->ofmap;
  lmem.fmt = CVK_FMT_I8;
  lmem.shape.h *= lmem.shape.w;
  lmem.shape.w = 1;
  lmem.stride = cvk_ctx->ops->tl_default_stride(cvk_ctx, lmem.shape, CVK_FMT_I8, 1);
  lmem.stride.h *= 2;
  p10.dst = &lmem;
  p10.src = mask->ifmap;
  p10.mv_lut_idx = true;
  cvk_ctx->ops->tdma_l2l_bf16_tensor_copy(cvk_ctx, &p10);
  p10.mv_lut_idx = false;

  cvk_tiu_lookup_table_param_t p12;
  p12.ofmap = mask->ofmap;
  p12.ifmap = mask->ifmap;
  p12.table = mask->pos_neg_table;
  cvk_ctx->ops->tiu_lookup_table(cvk_ctx, &p12);
}

inline void QuantizeMultiplierSmallerThanOne(float real_multiplier, uint32_t *quantized_multiplier,
                                             int *right_shift) {
  if (real_multiplier <= 0.f || real_multiplier > 1.f) {
    std::cerr << "Multiplier should be bigger than 0, smaller or euqal to 1." << std::endl;
    *quantized_multiplier = 0;
    *right_shift = 0;
    return;
  } else if (real_multiplier == 1.f) {
    *quantized_multiplier = (uint32_t)(1ll << 31) - 1;
    *right_shift = 0;
  } else {
    int s = 0;
    // We want to bring the real multiplier into the interval [1/2, 1).
    // We can do so by multiplying it by two, and recording how many times
    // we multiplied by two so that we can compensate that by a right
    // shift by the same amount.
    while (real_multiplier < 0.5f) {
      real_multiplier *= 2.0f;
      s++;
    }
    // Now that the real multiplier is in [1/2, 1), we convert it
    // into a fixed-point number.
    int64_t q = static_cast<int64_t>(round(real_multiplier * (1ll << 31)));
    assert(q <= (1ll << 31));
    // Handle the special case when the real multiplier was so close to 1
    // that its fixed-point approximation was undistinguishable from 1.
    // We handle this by dividing it by two, and remembering to decrement
    // the right shift amount.
    if (q == (1ll << 31)) {
      q /= 2;
      s--;
    }
    assert(s >= 0);
    assert(q <= (int64_t)LONG_MAX);
    *quantized_multiplier = (uint32_t)q;
    *right_shift = s;
  }
}

inline void pack_per_chan_cal_data(uint32_t channels, bool has_bias, int32_t *bias,
                                   uint32_t *multiplier, int8_t *shift, uint8_t *packed_data) {
  uint8_t *ptr = packed_data;

  for (uint32_t i = 0; i < channels; i++) {
    if (has_bias) {
      uint32_t val = (uint32_t)bias[i];
      *ptr = val & 0xff;
      ptr++;
      *ptr = (val >> 8) & 0xff;
      ptr++;
      *ptr = (val >> 16) & 0xff;
      ptr++;
      *ptr = (val >> 24) & 0xff;
      ptr++;
    }

    {
      uint32_t val = multiplier[i];
      *ptr = val & 0xff;
      ptr++;
      *ptr = (val >> 8) & 0xff;
      ptr++;
      *ptr = (val >> 16) & 0xff;
      ptr++;
      *ptr = (val >> 24) & 0xff;
      ptr++;
    }

    {
      uint8_t val = shift[i];
      *ptr = val;
      ptr++;
    }
  }
}

inline void getPackedMultiplierArrayBuffer(const uint32_t c, const uint32_t &quantized_multiplier,
                                           const int &right_shift, uint8_t *cal_data) {
  // Create tl_multiplier
  uint32_t *multiplier_data = new uint32_t[c];
  int8_t *shift_data = new int8_t[c];
  for (unsigned int i = 0; i < c; ++i) {
    // multipliers typically range in [2^30 ; 2^31 - 1].
    // Values in [0, 2^30 - 1] are normally unused, but harmless.
    // Thus a good way to randomize multipliers is to subtract from them
    // a random value smaller than 2^30 but still significant compared to it.
    multiplier_data[i] = quantized_multiplier;

    // Our H/W only supports right shift
    shift_data[i] = right_shift > 0 ? right_shift : 0;

#ifdef ENABLE_DEBUG_MSG
    printf("      [oc=%d] multiplier_data %d, shift_data %d\n", i, p_param->multiplier_data[i],
           p_param->shift_data[i]);
#endif
  }

  pack_per_chan_cal_data(c, 0, NULL, multiplier_data, shift_data, cal_data);
  delete[] multiplier_data;
  delete[] shift_data;
}

inline uint8_t *getPackedMultiplierArray(const uint32_t c, const uint32_t &quantized_multiplier,
                                         const int &right_shift) {
  const int per_chan_cal_data_size = MULTIPLIER_ONLY_PACKED_DATA_SIZE;  // p_param->has_bias ? 9 :
                                                                        // 5;  // bias(4) +
                                                                        // multiplier(4) + shift(1)
  const int cal_data_size = c * per_chan_cal_data_size;
  uint8_t *cal_data = (uint8_t *)malloc(cal_data_size);
  getPackedMultiplierArrayBuffer(c, quantized_multiplier, right_shift, cal_data);
  return cal_data;
}

static inline bmmem_device_t get_tensor_l2g(bmctx_t *ctx, cvk_context_t *cvk_ctx,
                                            const cvk_tl_t *tl) {
  cvk_tg_shape_t s;
  s.n = tl->shape.n;
  s.c = tl->shape.h;
  s.h = tl->shape.w;
  s.w = tl->shape.c;
  size_t total_size = s.n * s.c * s.h * s.w * getFmtSize(tl->fmt);
  cvk_tg_t tg;
  bmmem_device_t bm_dev = bmmem_device_alloc_raw(*ctx, total_size);
  tg.base_reg_index = 0;
  tg.start_address = bmmem_device_addr(bm_dev);
  tg.fmt = tl->fmt;
  tg.shape = s;
  tg.stride = cvk_ctx->ops->tg_default_stride(cvk_ctx, s, tl->fmt);
  cvk_tdma_l2g_tensor_copy_param_t p;
  p.src = tl;
  p.dst = &tg;
  cvk_ctx->ops->tdma_l2g_bf16_tensor_copy(cvk_ctx, &p);
  return bm_dev;
}

static inline uint8_t *get_bm_vaddr(bmctx_t *ctx, bmmem_device_t bm_dev) {
  if (bmmem_device_invld(*ctx, bm_dev) != BM_SUCCESS) {
    return nullptr;
  }
  return bmmem_device_v_addr(bm_dev);
}

// static inline uint8_t *get_tensor_l2g_submit(bmctx_t *ctx, cvk_context_t *cvk_ctx,
//                                         const cvk_tl_t *tl) {
//   bmmem_device_t bm_dev = get_tensor_l2g(ctx, cvk_ctx, tl);
//   cviruntime_cvikernel_submit(*ctx);
//   if (bmmem_device_invld(*ctx, bm_dev) != BM_SUCCESS) {
//     return nullptr;
//   }
//   uint8_t *bm_data = bmmem_device_v_addr(bm_dev);
//   size_t total_size = tl->shape.n * tl->shape.c * tl->shape.h * tl->shape.w *
//   getFmtSize(tl->fmt); uint8_t *data = new uint8_t[total_size]; memset(data, 1, total_size);
//   memcpy(data, bm_data, total_size); bmmem_device_free(*ctx, bm_dev); return data;
// }
