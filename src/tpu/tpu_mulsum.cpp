#include "tpu/tpu_mulsum.hpp"
#include <string.h>

int IveTPUMulSum::init(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx) {
  m_cmdbuf_subfix = "mulsum";
  m_force_use_ext = true;
  m_slice_info.io_fmt = FMT_BF16;
  m_slice_info.ping_pong_size = 1;
  m_slice_info.ping_pong_share_tl = 0;
  m_slice_info.nums_of_tl = 2 * 2;  // BF16

  return CVI_SUCCESS;
}

int IveTPUMulSum::runSetup(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx,
                           const std::vector<bmk1880v2_tensor_tgmem_shape_t> &tg_in_slices,
                           const std::vector<bmk1880v2_tensor_tgmem_shape_t> &tg_out_slices,
                           std::vector<u32> *tl_in_idx, std::vector<u32> *tl_out_idx,
                           const bool enable_cext) {
  m_input.clear();
  bmk1880v2_tensor_lmem_shape_t tl_shape;
  tl_shape.n = tg_in_slices[0].n;
  tl_shape.c = tg_in_slices[0].c;
  tl_shape.h = tg_in_slices[0].h;
  tl_shape.w = tg_in_slices[0].w;
  for (size_t i = 0; i < m_slice_info.ping_pong_size; i++) {
    m_input.emplace_back(allocTLMem(bk_ctx, tl_shape, FMT_BF16, 1));
  }
  mp_tl_mulsum = allocTLMem(bk_ctx, tl_shape, FMT_BF16, 1);
  m_tl_mulsum_shape = tl_shape;
  m_tl_mulsum_stride = mp_tl_mulsum->stride;
  constantFillTL(ctx, bk_ctx, convert_fp32_bf16(1.f), mp_tl_mulsum);

  m_p_mul.b_is_const = 0;
  m_p_mul.b = mp_tl_mulsum;
  m_p_mul.bf16_enable = 1;
  m_p_mul.relu_enable = 0;
  m_p_mul.rshift_bits = 0;
  m_p_mul.res_high = NULL;

  for (size_t pp = 0; pp < m_slice_info.ping_pong_size; pp++) {
    tl_in_idx->push_back(0 + pp);
  }
  return CVI_SUCCESS;
}

void IveTPUMulSum::operation(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx, u32 ping_idx) {
  m_p_mul.a = m_input[ping_idx];
  m_p_mul.res_low = mp_tl_mulsum;
  bmk1880v2_tiu_bf16_element_wise_mul(bk_ctx, &m_p_mul);
}

void IveTPUMulSum::beforeSubmit(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx,
                                std::vector<CviImg> &input, std::vector<CviImg> *output) {
  u32 total_data_size = m_tl_mulsum_shape.h * m_tl_mulsum_shape.w;
  u32 data_size = total_data_size;
  u32 fmt_size = getFmtSize(mp_tl_mulsum->fmt);
  bmk1880v2_tiu_element_wise_mul_param_t p_mul;
  bmk1880v2_tensor_lmem_t tl_1;
  bmk1880v2_tensor_lmem_t tl_2;
  tl_1.fmt = mp_tl_mulsum->fmt;
  tl_2.fmt = mp_tl_mulsum->fmt;
  while (data_size > 1) {
    u32 start_addr = mp_tl_mulsum->start_address;
    bool add_1 = false;
    if (data_size % 2 != 0) {
      add_1 = true;
      data_size -= 1;
      start_addr += fmt_size;
    }
    data_size /= 2;
    u32 w = data_size;
    u32 h = 1;
    auto m = w / 2;
    for (size_t i = 2; i < m; i++) {
      if (data_size % i == 0) {
        w = data_size / i;
        h = i;
        if (w < 4063) {
          break;
        }
      }
    }
    tl_1.start_address = start_addr;
    tl_2.start_address = start_addr + (h * w * fmt_size);
    tl_1.shape = {1, m_tl_mulsum_shape.c, h, w};
    tl_1.stride = bmk1880v2_bf16_tensor_lmem_default_stride(bk_ctx, tl_1.shape, tl_1.fmt, 1);
    tl_2.shape = tl_1.shape;
    tl_2.stride = tl_1.stride;
    p_mul.a = &tl_1;
    p_mul.b = &tl_2;
    p_mul.res_low = &tl_1;
    p_mul.res_high = NULL;
    p_mul.b_is_const = 0;
    p_mul.rshift_bits = 0;
    p_mul.bf16_enable = 1;
    p_mul.relu_enable = 0;
    bmk1880v2_tiu_bf16_element_wise_mul(bk_ctx, &p_mul);
    if (add_1) {
      data_size += 1;
    }
  }
  mp_tl_mulsum->shape = {m_tl_mulsum_shape.n, m_tl_mulsum_shape.c, 1, 1};
  mp_tl_mulsum->stride =
      bmk1880v2_bf16_tensor_lmem_default_stride(bk_ctx, mp_tl_mulsum->shape, mp_tl_mulsum->fmt, 1);
  m_sum = 1.f;
  m_bm_dev = get_tensor_l2g(ctx, bk_ctx, mp_tl_mulsum);
}

int IveTPUMulSum::postProcess(bmctx_t *ctx) {
  u8 *data = get_bm_vaddr(ctx, m_bm_dev);
  if (data == nullptr) {
    return CVI_FAILURE;
  }
  u16 *bf16_data = (u16 *)data;
  size_t total_size = m_tl_mulsum_shape.c;
  for (size_t i = 0; i < total_size; i++) {
    float val = convert_bf16_fp32(bf16_data[i]);
    m_sum *= val;
  }
  bmmem_device_free(*ctx, m_bm_dev);
  m_bm_dev = NULL;
  return CVI_SUCCESS;
}

double IveTPUMulSum::getSum() { return m_sum; }