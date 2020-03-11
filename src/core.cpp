#include "core.hpp"

#include <cmath>
#include <iostream>
#include "debug.hpp"

inline void GetSliceUnitProperty(const u32 length, const u32 slice, const int kernel_sz,
                                 const int default_stride, u32 pad_0, u32 pad_1, sliceUnit *unit) {
  unit->slice = slice > length ? length : slice;
  unit->slice = default_stride * (int)(unit->slice / default_stride);
  unit->skip = unit->slice - kernel_sz + default_stride;
  unit->skip = (u32)default_stride > unit->skip ? default_stride : unit->skip;

  u32 &&pad_total = pad_0 + pad_1;
  unit->turn = ((int64_t)length - unit->slice - pad_1) / unit->skip + 1;
  int64_t result = (int64_t)length - (int64_t)((unit->turn) * (unit->slice - pad_total));
  if (result >= kernel_sz) {
    // x + (x - 1) * (kz - (pad_tatal - stride + 1))
    int res_left = result - kernel_sz;
    int kz_unit = (kernel_sz - (pad_total - default_stride + 1));
    int res_div = res_left / kz_unit;
    int result = kz_unit * res_div + kernel_sz;
    unit->left = result;
    unit->turn++;
  } else if (result < 0 && unit->slice > std::abs(result)) {
    unit->left = unit->slice + result;
  } else {
    unit->left = 0;
  }
}

inline void categoryIOTLShape(
    const std::vector<bmk1880v2_tensor_lmem_t *> &tl_vec,
    std::vector<bmk1880v2_tensor_tgmem_shape_t> &s_in_vec,
    std::vector<bmk1880v2_tensor_tgmem_shape_t> &s_out_vec,
    std::vector<std::pair<int, bmk1880v2_tensor_lmem_t *>> *tl_in_shape_lmem_vec,
    std::vector<std::pair<int, bmk1880v2_tensor_lmem_t *>> *tl_out_shape_lmem_vec) {
  tl_in_shape_lmem_vec->clear();
  tl_out_shape_lmem_vec->clear();
  for (size_t i = 0; i < tl_vec.size(); i++) {
    auto *lmem = tl_vec[i];
    bool skip = false;
    for (size_t k = 0; k < s_in_vec.size(); k++) {
      if (tgTLShapeCompare(lmem->shape, s_in_vec[k])) {
        tl_in_shape_lmem_vec->push_back({k, lmem});
        skip = true;
        break;
      }
    }
    if (skip) {
      continue;
    }
    for (size_t k = 0; k < s_out_vec.size(); k++) {
      if (tgTLShapeCompare(lmem->shape, s_out_vec[k])) {
        tl_out_shape_lmem_vec->push_back({k, lmem});
        break;
      }
    }
  }
}

inline void getTLInfo(const std::vector<bmk1880v2_tensor_lmem_t *> &tl_vec,
                      const std::vector<u32> &tl_in_idx, const std::vector<u32> &tl_out_idx,
                      TLInfo *tl_in_info, TLInfo *tl_out_info) {
  for (size_t i = 0; i < tl_in_idx.size(); i++) {
    auto *lmem = tl_vec[tl_in_idx[i]];
    tl_in_info->lmem_vec.emplace_back(lmem);
    tl_in_info->fns_vec.push_back(FmtnSize(lmem->fmt));
  }
  for (size_t i = 0; i < tl_out_idx.size(); i++) {
    auto *lmem = tl_vec[tl_out_idx[i]];
    tl_out_info->lmem_vec.emplace_back(lmem);
    tl_out_info->fns_vec.push_back(FmtnSize(lmem->fmt));
  }
}

inline void getBMAddrInfo(const std::vector<CviImg> &input, const std::vector<CviImg> &output,
                          const int pad_left, const int pad_top, BMAddrInfo *bm_src_info,
                          BMAddrInfo *bm_dest_info) {
  for (size_t k = 0; k < input.size(); k++) {
    u64 bm_start_addr = input[k].GetPAddr();
    bm_src_info->addr_vec.push_back(bm_start_addr);
    bm_src_info->fns_vec.push_back(FmtnSize(input[k].m_tg.fmt));
  }
  for (size_t k = 0; k < output.size(); k++) {
    u64 bm_des_addr = output[k].GetPAddr();
    FmtnSize fns(output[k].m_tg.fmt);
    u64 new_bm_des_addr =
        bm_des_addr + (output[k].m_tg.stride.h * pad_top) + (pad_left * fns.getSize());
    bm_dest_info->addr_vec.push_back(new_bm_des_addr);
    bm_dest_info->fns_vec.push_back(fns);
  }
}

inline int checkIsBufferOverflow(const std::vector<CviImg> &input,
                                 const std::vector<CviImg> &output, const BMAddrInfo &bm_src_info,
                                 const BMAddrInfo &bm_dest_info, const int &pad_l, const int &pad_t,
                                 const bool is_1d) {
#if DISABLE_OVERFLOWCHECK
  return BM_SUCCESS;
#else
  int ret = BM_SUCCESS;
  for (size_t k = 0; k < input.size(); k++) {
    const u64 bm_start_addr = input[k].GetPAddr();
    u64 jumped_value = bm_src_info.addr_vec[k] - bm_start_addr;
    u32 total_addr = is_1d ? input[k].m_tg.stride.n : input[k].m_tg.stride.c;
    if (jumped_value != total_addr) {
      printf(
          "Error! Input %lu jumped value not align to image size %lu, original %u, start addr "
          "%lu\n",
          k, jumped_value, total_addr, bm_start_addr);
      ret = BM_ERR_FAILURE;
    }
  }
  for (size_t k = 0; k < output.size(); k++) {
    const u64 bm_des_addr = output[k].GetPAddr();
    u64 jumped_value = bm_dest_info.addr_vec[k] - bm_des_addr;
    u32 total_addr =
        is_1d ? output[k].m_tg.stride.n
              : (output[k].m_tg.stride.c +
                 ((output[k].m_tg.stride.h * pad_t) + (pad_l * bm_dest_info.fns_vec[k].getSize())));
    if (jumped_value != total_addr) {
      printf(
          "Error! Output %lu jumped value not align to image size %lu, original %u, start addr "
          "%lu\n",
          k, jumped_value, total_addr, bm_des_addr);
      ret = BM_ERR_FAILURE;
    }
  }
  return ret;
#endif
}

// For channel Ext mode
inline int channelExtension(bmk1880v2_context_t *bk_ctx, const int ic, const int ih, const int iw,
                            const int h_cext_multiplier, const int pad_left, const int pad_right,
                            const int pad_top, const int pad_bottom, const int kh, const int kw,
                            const int k_stride_h, const int k_stride_w, fmt_t fmt_type,
                            TensorSliceInfo *tsi) {
  bmk1880v2_tensor_lmem_shape_t tl_weight_shape;
  bmk1880v2_tensor_lmem_shape_t tl_bias_shape;

  if (bm1880v2_reshape_channel_same(
          bk_ctx, ic, ih, iw, kh, kw, pad_right, pad_left, k_stride_h, k_stride_w,
          &tsi->tl_load.shape, &tsi->tl_load.stride, &tsi->tg_load.shape, &tsi->tg_load.stride,
          &tl_weight_shape, &tl_bias_shape, &tsi->tl_store.shape, fmt_type, CTRL_AL) == -1) {
    std::cerr << "Extend failed." << std::endl;
    return BM_ERR_FAILURE;
  }
  tsi->tg_store.shape.n = tsi->tl_store.shape.n;
  tsi->tg_store.shape.c = tsi->tl_store.shape.c;
  tsi->tg_store.shape.h = tsi->tl_store.shape.h;
  tsi->tg_store.shape.w = tsi->tl_store.shape.w;
  tsi->tg_store.stride = bmk1880v2_bf16_tensor_tgmem_default_stride(tsi->tg_store.shape, fmt_type);
  tsi->tl_store.stride =
      bmk1880v2_bf16_tensor_lmem_default_stride(bk_ctx, tsi->tl_store.shape, 1, fmt_type);
  // clang-format off
  IVE_DEBUG("[Summary]\n" \
            " Original shape: n c h w %d %d %d %d\n" \
            " Kernel size: h w %d %d\n" \
            " Pad: left right %d %d\n" \
            " Kernel stride: h w %d %d\n" \
            " => tl load shape %d %d %d %d\n" \
            " => tl load stride %d %d %d %d\n" \
            " => tg load shape %d %d %d %d\n" \
            " => tg load stride %d %d %d\n" \
            " => tl store shape %d %d %d %d\n" \
            " => tl store stride %d %d %d %d\n",
            1, ic, ih, iw,
            kh, kw,
            pad_left, pad_right,
            k_stride_h, k_stride_w,
            tsi->tl_load.shape.n, tsi->tl_load.shape.c, tsi->tl_load.shape.h, tsi->tl_load.shape.w,
            tsi->tl_load.stride.n, tsi->tl_load.stride.c, tsi->tl_load.stride.h, tsi->tl_load.stride.w,
            tsi->tg_load.shape.n, tsi->tg_load.shape.c, tsi->tg_load.shape.h, tsi->tg_load.shape.w,
            tsi->tg_load.stride.n, tsi->tg_load.stride.c, tsi->tg_load.stride.h,
            tsi->tl_store.shape.n, tsi->tl_store.shape.c, tsi->tl_store.shape.h, tsi->tl_store.shape.w,
            tsi->tl_store.stride.n, tsi->tl_store.stride.c, tsi->tl_store.stride.h, tsi->tl_store.stride.w);
  // clang-format on
  u32 h_input_single_lane = ih / h_cext_multiplier + pad_bottom + pad_top;
  if (tsi->tg_load.shape.h != h_input_single_lane) {
    std::cerr << "H extend c multiplier: " << h_cext_multiplier << std::endl;
    std::cerr << "Predicted h_slice not match. " << tsi->tg_load.shape.h << ", "
              << h_input_single_lane << std::endl;
    return BM_ERR_FAILURE;
  }
  return BM_SUCCESS;
}

inline bool updateIfExceedTLMem(const int &npu_num, const u32 &c_multiplier, const int &img_c,
                                const u32 &out_slice_max, const u32 &out_slice_left,
                                const u32 &k_stride, u32 *new_c_multiplier,
                                u32 *out_h_left_pixels) {
  u32 tmp_h = out_slice_left / c_multiplier;
  u32 &new_c_multiplier_a = *new_c_multiplier;
  if (tmp_h > out_slice_max) {
    u32 unit_skip = out_slice_max + (k_stride - 1);
    new_c_multiplier_a = npu_num / img_c;
    tmp_h = new_c_multiplier_a * unit_skip;
    while (tmp_h > out_slice_left) {
      new_c_multiplier_a--;
      tmp_h = new_c_multiplier_a * unit_skip;
    }
    u32 new_h_left = out_slice_max * new_c_multiplier_a;
    *out_h_left_pixels = out_slice_left - new_h_left;
    return true;
  } else {
    new_c_multiplier_a = c_multiplier;
    *out_h_left_pixels = 0;
  }
  return false;
}

inline void updateTSIInfo(bmk1880v2_context_t *bk_ctx, const u32 load_n, const u32 load_c,
                          const u32 load_h, const u32 load_w, const u32 store_n, const u32 store_c,
                          const u32 store_h, const u32 store_w, const fmt_t io_fmt,
                          const fmt_t img_fmt, TensorSliceInfo *tl_info) {
  tl_info->tl_load.shape = {load_n, load_c, load_h, load_w};
  tl_info->tl_load.stride =
      bmk1880v2_bf16_tensor_lmem_default_stride(bk_ctx, tl_info->tl_load.shape, 1, io_fmt);
  tl_info->tg_load.shape = {load_n, load_c, load_h, load_w};
  tl_info->tg_load.stride =
      bmk1880v2_bf16_tensor_tgmem_default_stride(tl_info->tg_load.shape, img_fmt);
  tl_info->tl_store.shape = {store_n, store_c, store_h, store_w};
  tl_info->tl_store.stride =
      bmk1880v2_bf16_tensor_lmem_default_stride(bk_ctx, tl_info->tl_store.shape, 1, io_fmt);
  tl_info->tg_store.shape = {store_n, store_c, store_h, store_w};
  tl_info->tg_store.stride =
      bmk1880v2_bf16_tensor_tgmem_default_stride(tl_info->tg_store.shape, img_fmt);
}

inline void updateLMemSize(
    bmk1880v2_context_t *bk_ctx, const int &npu_num, const fmt_t &io_fmt,
    const TensorSliceInfo &tsi,
    std::vector<std::pair<int, bmk1880v2_tensor_lmem_t *>> *tl_in_shape_lmem_vec,
    std::vector<std::pair<int, bmk1880v2_tensor_lmem_t *>> *tl_out_shape_lmem_vec,
    std::vector<bmk1880v2_tensor_lmem_t *> *m_tl_vec) {
  for (size_t k = 0; k < tl_in_shape_lmem_vec->size(); k++) {
    int &index = (*tl_in_shape_lmem_vec)[k].first;
    auto *lmem = (*tl_in_shape_lmem_vec)[k].second;
    lmem->shape = tsi.tl_load.shape;
    if (io_fmt == lmem->fmt) {
      lmem->stride = tsi.tl_load.stride;
    } else {
      lmem->stride = bmk1880v2_bf16_tensor_lmem_default_stride(bk_ctx, lmem->shape, 1, lmem->fmt);
    }
  }
  for (size_t k = 0; k < tl_out_shape_lmem_vec->size(); k++) {
    int &index = (*tl_out_shape_lmem_vec)[k].first;
    auto *lmem = (*tl_out_shape_lmem_vec)[k].second;
    lmem->shape = tsi.tl_store.shape;
    if (io_fmt == lmem->fmt) {
      lmem->stride = tsi.tl_store.stride;
    } else {
      lmem->stride = bmk1880v2_bf16_tensor_lmem_default_stride(bk_ctx, lmem->shape, 1, lmem->fmt);
    }
  }
  for (size_t k = 0; k < m_tl_vec->size(); k++) {
    auto *lmem = (*m_tl_vec)[k];
    u64 &&align_up_res = align_up(lmem->shape.c, npu_num) / npu_num;
    int is_align = (lmem->stride.n != align_up_res) ? 1 : 0;
    if (lmem->shape.c != tsi.tl_load.shape.c) {
      lmem->shape.c = tsi.tl_load.shape.c;
      lmem->stride =
          bmk1880v2_bf16_tensor_lmem_default_stride(bk_ctx, lmem->shape, is_align, lmem->fmt);
    }
  }
}
// Ext end

IveCore::IveCore() {
  auto chip_info = bmk1880v2_chip_info();
  m_chip_info.eu_num = chip_info.eu_num;
  m_chip_info.lmem_bank_size = chip_info.lmem_bank_size;
  m_chip_info.lmem_banks = chip_info.lmem_banks;
  m_chip_info.lmem_size = chip_info.lmem_size;
  m_chip_info.npu_num = chip_info.npu_num;
  m_chip_info.version = chip_info.version;
}

int IveCore::getSlice(const u32 nums_of_lmem, const u32 nums_of_table, const u32 fixed_lmem_size,
                      const u32 n, const u32 c, const u32 h, const u32 w, const u32 table_size,
                      const kernelInfo kernel_info, const int npu_num, sliceUnit *unit_h,
                      sliceUnit *unit_w, const bool enable_cext) {
  if (c > 32) {
    std::cerr << "Channel exceed limitation." << std::endl;
    return BM_ERR_FAILURE;
  }
  // Calculate fixed kernel size
  u32 kernel_sz = (kernel_info.nums_of_kernel * kernel_info.size * kernel_info.size +
                   MULTIPLIER_ONLY_PACKED_DATA_SIZE * kernel_info.use_multiplier);
  // Find max available mem for one tl.
  int64_t result = m_chip_info.lmem_size - (int64_t)(kernel_sz + table_size * nums_of_table) -
                   (int64_t)fixed_lmem_size;
  if (result < 0) {
    std::cerr << "Insufficient memory: " << result << std::endl;
    return BM_ERR_FAILURE;
  }
  const u32 available_lmem_per_tl = (u32)result / nums_of_lmem;
  u32 w_length = w;
  u32 h_tmp_slice = 0;
  int w_num = 1;
  // Here the default value for kernel size is 1. The h_slice should never smaller than kernel size.
  h_tmp_slice = available_lmem_per_tl / w_length;
  w_num++;
  if (kernel_info.size == 1) {
    while (h_tmp_slice < kernel_info.size) {
      w_length = w / w_num;
      u32 h_tmp_res = available_lmem_per_tl / w_length;
      h_tmp_slice = h_tmp_res > h ? h : h_tmp_res;
      w_num++;
    }
  } else {
    while (h_tmp_slice < kernel_info.size) {
      w_length = w / w_num;
      if (w_length + 16 < w) {
        int res = w_length % 16;
        w_length += (res == 0 ? 16 : res);
      }
      u32 h_tmp_res = available_lmem_per_tl / w_length;
      h_tmp_slice = h_tmp_res > h ? h : h_tmp_res;
      w_num++;
    }
  }
  // If enable channel extending feature
  if (enable_cext) {
    // FIXME: Pass eu num here
    int c_multiplier = (int)(npu_num / c) - 1;
    u32 unit_skip =
        h_tmp_slice - kernel_info.pad[2] - kernel_info.pad[3] + (kernel_info.default_stride_y - 1);
    h_tmp_slice += unit_skip * c_multiplier;
    while (h_tmp_slice > h) {
      h_tmp_slice -= unit_skip;
      c_multiplier--;
    }
    unit_h->c_multiplier = c_multiplier;
  }

  // FIXME: Logic error
  GetSliceUnitProperty(h, h_tmp_slice, kernel_info.size, kernel_info.default_stride_y,
                       kernel_info.pad[2], kernel_info.pad[3], unit_h);
  GetSliceUnitProperty(w, w_length, kernel_info.size, kernel_info.default_stride_x,
                       kernel_info.pad[0], kernel_info.pad[1], unit_w);
  IVE_DEBUG("H slice %d skip %d turn %d left %d\n", unit_h->slice, unit_h->skip, unit_h->turn,
            unit_h->left);
  IVE_DEBUG("W slice %d skip %d turn %d left %d\n", unit_w->slice, unit_w->skip, unit_w->turn,
            unit_w->left);
  return BM_SUCCESS;
}

bmk1880v2_tensor_lmem_t *IveCore::allocTLMem(bmk1880v2_context_t *bk_ctx,
                                             bmk1880v2_tensor_lmem_shape_t tl_shape, fmt_t fmt,
                                             int eu_align) {
  bmk1880v2_tensor_lmem_t *lmem = bmk1880v2_lmem_alloc_bf16_tensor(bk_ctx, tl_shape, fmt, eu_align);
  if (lmem == NULL) {
    std::cerr << "Tensor allocate failed. Shape: " << tl_shape.n << ", " << tl_shape.c << ", "
              << tl_shape.h << ", " << tl_shape.w << std::endl;
    return nullptr;
  }
  m_tl_vec.emplace_back(lmem);
  return lmem;
}

int IveCore::freeTLMems(bmk1880v2_context_t *bk_ctx) {
  for (int i = m_tl_vec.size() - 1; i >= 0; i--) {
    bmk1880v2_lmem_free_tensor(bk_ctx, m_tl_vec[i]);
  }
  m_tl_vec.clear();
  return BM_SUCCESS;
}

int IveCore::sliceSetup(SliceRes &slice_res, SliceRes *tg_in_res, SliceRes *tg_out_res) {
  *tg_in_res = slice_res;
  *tg_out_res = slice_res;
  return BM_SUCCESS;
}

int IveCore::freeChildTGMem(bmctx_t *ctx) { return BM_SUCCESS; }

int IveCore::runSingleSizeKernel(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx,
                                 std::vector<CviImg> &input, std::vector<CviImg> *output,
                                 bool enable_min_max) {
  // FIXME: Support later
  if (m_slice_info.ping_pong_size != 1) {
    std::cerr << "Currently runSingleSizeKernel does not support ping pong." << std::endl;
    m_slice_info.ping_pong_size = 1;
  }
  u32 batch = input[0].m_tg.shape.n;
  u32 channel = input[0].m_tg.shape.c;
  u32 height = input[0].m_tg.shape.h;
  u32 width = input[0].m_tg.shape.w;
  std::vector<bool> find_min_max;
  // Insert extra tl
  u32 nums_of_tl = m_slice_info.nums_of_tl;
  u32 fix_lmem_size = m_slice_info.fix_lmem_size;
#if 0  // Disable now
  if (enable_min_max) {
    nums_of_tl += 1;
    fix_lmem_size += (2 * (*output)[0].m_tg.shape.c);
  }
  for (size_t i = 0; i < output->size(); i++) {
    find_min_max.emplace_back(enable_min_max);
    if (enable_min_max) {
      nums_of_tl += 1;
      fix_lmem_size += (2 * 2 * (*output)[i].m_tg.shape.c);
    }
  }
#endif
  // FIXME: Move to constructor if possible.
  bmk1880v2_tensor_lmem_shape_t tl_table_s;
  u64 result = bf16_lut_tbl_bytesize(bk_ctx, &tl_table_s, FMT_U8);
  m_table_per_channel_size = result / m_chip_info.npu_num;  // 32 * 8 for bm1880v2
  SliceRes slice_res;
  int ret = getSlice(nums_of_tl, m_slice_info.nums_of_table, fix_lmem_size, batch, channel, height,
                     width, m_table_per_channel_size, m_kernel_info, m_chip_info.npu_num,
                     &slice_res.h, &slice_res.w, false);
  if (ret != BM_SUCCESS) {
    return BM_ERR_FAILURE;
  }

  SliceRes in_slice_res, out_slice_res;
  sliceSetup(slice_res, &in_slice_res, &out_slice_res);
  if (in_slice_res.h.turn != out_slice_res.h.turn) {
    std::cerr << "Input/ output h slice turn are not the same " << in_slice_res.h.turn << ", "
              << out_slice_res.h.turn << std::endl;
  }
  if (in_slice_res.w.turn != out_slice_res.w.turn) {
    std::cerr << "Input/ output w slice turn are not the same " << in_slice_res.w.turn << ", "
              << out_slice_res.w.turn << std::endl;
  }

  // Setup slice input/ output shapes and left shapes
  std::vector<bmk1880v2_tensor_tgmem_shape_t> s_in_vec, s_out_vec;
  for (size_t k = 0; k < input.size(); k++) {
    s_in_vec.push_back({1, channel, in_slice_res.h.slice, in_slice_res.w.slice});
  }
  for (size_t k = 0; k < output->size(); k++) {
    s_out_vec.push_back({1, channel, out_slice_res.h.slice, out_slice_res.w.slice});
  }
  std::vector<bmk1880v2_tensor_tgmem_shape_t> s_in_left_vec, s_out_left_vec;
  for (size_t k = 0; k < input.size(); k++) {
    s_in_left_vec.push_back({1, channel, in_slice_res.h.left, in_slice_res.w.left});
  }
  for (size_t k = 0; k < output->size(); k++) {
    s_out_left_vec.push_back({1, channel, out_slice_res.h.left, out_slice_res.w.left});
  }

  // allocate tl shape and get input/ output indices.
  std::vector<u32> tl_in_idx, tl_out_idx;
  runSetup(ctx, bk_ctx, s_in_vec, s_out_vec, &tl_in_idx, &tl_out_idx, false);

  // Dummy check, can be turned off in official release
  if (tl_in_idx.size() != input.size()) {
    std::cerr << "Input tl size not match input image num " << tl_in_idx.size() << ", "
              << input.size() << std::endl;
  }
  if (tl_out_idx.size() != output->size()) {
    std::cerr << "Output tl size not match input image num " << tl_out_idx.size() << ", "
              << output->size() << std::endl;
  }
  // Dummy check end
  // Category tl shapes
  std::vector<std::pair<int, bmk1880v2_tensor_lmem_t *>> tl_in_shape_lmem_vec,
      tl_out_shape_lmem_vec;
  categoryIOTLShape(m_tl_vec, s_in_vec, s_out_vec, &tl_in_shape_lmem_vec, &tl_out_shape_lmem_vec);

  // Find and create input/ output fmt size pair.
  TLInfo tl_in_info, tl_out_info;
  getTLInfo(m_tl_vec, tl_in_idx, tl_out_idx, &tl_in_info, &tl_out_info);

  // Get device memory start offset
  BMAddrInfo bm_src_info, bm_dest_info;
  getBMAddrInfo(input, *output, m_kernel_info.pad[0], m_kernel_info.pad[2], &bm_src_info,
                &bm_dest_info);

  // Create tg block
  bmk1880v2_tensor_tgmem_t tg_in;
  tg_in.base_reg_index = 0;
  bmk1880v2_tensor_tgmem_t tg_out;
  tg_out.base_reg_index = 0;

  // Main for loop
  for (u32 i = 0; i < slice_res.h.turn; i++) {
    // Re-assign head address to w.
    std::vector<u64> bm_src_addr_w = bm_src_info.addr_vec;
    std::vector<u64> bm_dest_addr_w = bm_dest_info.addr_vec;
    // Change H TL size to fit left shape in last turn
    for (size_t k = 0; k < tl_in_shape_lmem_vec.size(); k++) {
      int &index = tl_in_shape_lmem_vec[k].first;
      auto *lmem = tl_in_shape_lmem_vec[k].second;
      if (s_in_left_vec[index].h != 0) {
        if (i == 0) {
          lmem->shape.h = s_in_vec[index].h;
          lmem->stride =
              bmk1880v2_bf16_tensor_lmem_default_stride(bk_ctx, lmem->shape, 1, lmem->fmt);
        } else if (i == in_slice_res.h.turn - 1) {
          lmem->shape.h = s_in_left_vec[index].h;
          lmem->stride =
              bmk1880v2_bf16_tensor_lmem_default_stride(bk_ctx, lmem->shape, 1, lmem->fmt);
        }
      }
    }
    for (size_t k = 0; k < tl_out_shape_lmem_vec.size(); k++) {
      int &index = tl_out_shape_lmem_vec[k].first;
      auto *lmem = tl_out_shape_lmem_vec[k].second;
      if (s_out_left_vec[index].h != 0) {
        if (i == 0) {
          lmem->shape.h = s_out_vec[index].h;
          lmem->stride =
              bmk1880v2_bf16_tensor_lmem_default_stride(bk_ctx, lmem->shape, 1, lmem->fmt);
        } else if (i == out_slice_res.h.turn - 1) {
          lmem->shape.h = s_out_left_vec[index].h;
          lmem->stride =
              bmk1880v2_bf16_tensor_lmem_default_stride(bk_ctx, lmem->shape, 1, lmem->fmt);
        }
      }
    }

    for (u32 j = 0; j < slice_res.w.turn; j++) {
      // Change W TL size to fit left shape in last turn
      for (size_t k = 0; k < tl_in_shape_lmem_vec.size(); k++) {
        int &index = tl_in_shape_lmem_vec[k].first;
        auto *lmem = tl_in_shape_lmem_vec[k].second;
        if (s_in_left_vec[index].w != 0) {
          if (j == 0) {
            lmem->shape.w = s_in_vec[index].w;
            lmem->stride =
                bmk1880v2_bf16_tensor_lmem_default_stride(bk_ctx, lmem->shape, 1, lmem->fmt);
          } else if (j == in_slice_res.w.turn - 1) {
            lmem->shape.w = s_in_left_vec[index].w;
            lmem->stride =
                bmk1880v2_bf16_tensor_lmem_default_stride(bk_ctx, lmem->shape, 1, lmem->fmt);
          }
        }
      }
      for (size_t k = 0; k < tl_out_shape_lmem_vec.size(); k++) {
        int &index = tl_out_shape_lmem_vec[k].first;
        auto *lmem = tl_out_shape_lmem_vec[k].second;
        if (s_out_left_vec[index].w != 0) {
          if (j == 0) {
            lmem->shape.w = s_out_vec[index].w;
            lmem->stride =
                bmk1880v2_bf16_tensor_lmem_default_stride(bk_ctx, lmem->shape, 1, lmem->fmt);
          } else if (j == out_slice_res.w.turn - 1) {
            lmem->shape.w = s_out_left_vec[index].w;
            lmem->stride =
                bmk1880v2_bf16_tensor_lmem_default_stride(bk_ctx, lmem->shape, 1, lmem->fmt);
          }
        }
      }

      // tg2tl
      for (size_t k = 0; k < tl_in_info.lmem_vec.size(); k++) {
        tg_in.start_address = bm_src_addr_w[k];
        tg_in.shape.n = tl_in_info.lmem_vec[k]->shape.n;
        tg_in.shape.c = tl_in_info.lmem_vec[k]->shape.c;
        tg_in.shape.h = tl_in_info.lmem_vec[k]->shape.h;
        tg_in.shape.w = tl_in_info.lmem_vec[k]->shape.w;
        tg_in.fmt = bm_src_info.fns_vec[k].getFmt();
        tg_in.stride = input[k].m_tg.stride;
        bmk1880v2_tdma_tg2l_tensor_copy_param_t p_copy_in;
        p_copy_in.src = &tg_in;
        p_copy_in.dst = tl_in_info.lmem_vec[k];
        bmk1880v2_tdma_g2l_bf16_tensor_copy(bk_ctx, &p_copy_in);

        // Change src head addr
        bm_src_addr_w[k] += 1 * in_slice_res.w.skip * bm_src_info.fns_vec[k].getSize();
      }

      operation(ctx, bk_ctx, 0);

      // tl2tg
      for (size_t k = 0; k < tl_out_info.lmem_vec.size(); k++) {
        tg_out.start_address = bm_dest_addr_w[k];
        tg_out.fmt = bm_dest_info.fns_vec[k].getFmt();
        tg_out.shape.n = tl_out_info.lmem_vec[k]->shape.n;
        tg_out.shape.c = tl_out_info.lmem_vec[k]->shape.c;
        tg_out.shape.h =
            tl_out_info.lmem_vec[k]->shape.h - (m_kernel_info.pad[2] + m_kernel_info.pad[3]);
        tg_out.shape.w =
            tl_out_info.lmem_vec[k]->shape.w - (m_kernel_info.pad[0] + m_kernel_info.pad[1]);
        tg_out.stride = (*output)[k].m_tg.stride;
        bmk1880v2_tensor_lmem_t out_shape;
        auto &tl_out = tl_out_info.lmem_vec;
        // printf("st addr%d, tg st addr %lu\n", tl_out[k]->start_address, bm_dest_addr_w[k]);
        out_shape.start_address = tl_out[k]->start_address +
                                  (1 * tl_out[k]->stride.h * m_kernel_info.pad[2]) +
                                  (m_kernel_info.pad[0] * tl_out_info.fns_vec[k].getSize());
        out_shape.fmt = tl_out[k]->fmt;
        out_shape.cmprs_fmt = tl_out[k]->cmprs_fmt;
        out_shape.shape = tl_out[k]->shape;
        out_shape.shape.h = tg_out.shape.h;
        out_shape.shape.w = tg_out.shape.w;
        out_shape.stride = tl_out[k]->stride;
        bmk1880v2_tdma_l2tg_tensor_copy_param_t p_copy_out;
        p_copy_out.src = &out_shape;
        p_copy_out.dst = &tg_out;
        bmk1880v2_tdma_l2g_bf16_tensor_copy(bk_ctx, &p_copy_out);

        // Change dest head addr
        bm_dest_addr_w[k] += 1 * out_slice_res.w.skip * bm_dest_info.fns_vec[k].getSize();
      }
    }
    // Change src/ dest head addr
    for (size_t k = 0; k < bm_src_info.addr_vec.size(); k++) {
      auto jump_val = (i == in_slice_res.h.turn - 1 && in_slice_res.h.left != 0)
                          ? in_slice_res.h.left
                          : in_slice_res.h.skip;
      bm_src_info.addr_vec[k] += 1 * input[k].m_tg.stride.h * jump_val;
    }
    for (size_t k = 0; k < bm_dest_info.addr_vec.size(); k++) {
      auto jump_val = (i == out_slice_res.h.turn - 1 && out_slice_res.h.left != 0)
                          ? out_slice_res.h.left
                          : out_slice_res.h.skip;
      bm_dest_info.addr_vec[k] += 1 * (*output)[k].m_tg.stride.h * jump_val;
    }
  }
  IVE_DEBUG("Slice info:\n");
  IVE_DEBUG("{ h_slice, h_turn, h_skip, h_left} = { %d, %d, %d, %d}\n", in_slice_res.h.slice,
            in_slice_res.h.turn, in_slice_res.h.skip, in_slice_res.h.left);
  IVE_DEBUG("{ w_slice, w_turn, w_skip, w_left} = { %d, %d, %d, %d}\n", in_slice_res.w.slice,
            in_slice_res.w.turn, in_slice_res.w.skip, in_slice_res.w.left);

  // Dummy gaurd for buffer overflow
  ret |= checkIsBufferOverflow(input, *output, bm_src_info, bm_dest_info, m_kernel_info.pad[0],
                               m_kernel_info.pad[2], false);
  if (ret == BM_SUCCESS) {
    bmruntime_bmkernel_submit(*ctx);
  }
  freeTLMems(bk_ctx);
  freeChildTGMem(ctx);
  return ret;
}

int IveCore::runSingleSizeExtKernel(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx,
                                    std::vector<CviImg> &input, std::vector<CviImg> *output,
                                    bool enable_min_max) {
  if (m_slice_info.io_fmt == FMT_INVALID) {
    std::cerr << "Invalid fmt engine type." << std::endl;
    return BM_ERR_FAILURE;
  }
  // FIXME: Support later
  if (m_slice_info.ping_pong_size != 1) {
    std::cerr << "Currently runSingleSizeKernel does not support ping pong." << std::endl;
    m_slice_info.ping_pong_size = 1;
  }
  u32 batch = input[0].m_tg.shape.n;
  u32 channel = input[0].m_tg.shape.c;
  u32 height = input[0].m_tg.shape.h;
  u32 width = input[0].m_tg.shape.w;
  // Insert extra tl
  u32 nums_of_tl = m_slice_info.nums_of_tl;
  u32 fix_lmem_size = m_slice_info.fix_lmem_size;

  // FIXME: Move to constructor if possible.
  bmk1880v2_tensor_lmem_shape_t tl_table_s;
  u64 result = bf16_lut_tbl_bytesize(bk_ctx, &tl_table_s, FMT_U8);
  m_table_per_channel_size = result / m_chip_info.npu_num;  // 32 * 8 for bm1880v2
  SliceRes slice_res;
  int ret = getSlice(nums_of_tl, m_slice_info.nums_of_table, fix_lmem_size, batch, channel, height,
                     width, m_table_per_channel_size, m_kernel_info, m_chip_info.npu_num,
                     &slice_res.h, &slice_res.w, true);
  if (ret != BM_SUCCESS) {
    return BM_ERR_FAILURE;
  }
  // FIXME: Slice Setup not supported in this mode yet.
  // sliceSetup(slice_res, &in_slice_res, &out_slice_res);
  // channelExt will temporarily replace them.
  // Need mechanism for algorithm such as BLOCK
  SliceRes in_slice_res, out_slice_res;
  in_slice_res = slice_res;
  out_slice_res = slice_res;
  // Convert to API acceptance input.
  const int vertical_pad_total = m_kernel_info.pad[2] + m_kernel_info.pad[3];
  out_slice_res.h.slice -= vertical_pad_total;
  out_slice_res.h.left -= vertical_pad_total;
  // channel ext supports w, so no need to change out_slice_res.w
  // out_slice_res.w.slice -= (m_kernel_info.pad[0] + m_kernel_info.pad[1]);
  // out_slice_res.w.left -= (m_kernel_info.pad[0] + m_kernel_info.pad[1]);
  TensorSliceInfo out_info, out_w_info, out_h_info, out_wh_info;
  channelExtension(bk_ctx, channel, out_slice_res.h.slice, out_slice_res.w.slice,
                   out_slice_res.h.c_multiplier, m_kernel_info.pad[0], m_kernel_info.pad[1],
                   m_kernel_info.pad[2], m_kernel_info.pad[3], m_kernel_info.size,
                   m_kernel_info.size, m_kernel_info.default_stride_y,
                   m_kernel_info.default_stride_x, m_slice_info.io_fmt, &out_info);
  u32 out_slice_res_h_left_pixels = 0;
  u32 in_slice_res_h_left_pixels = 0;
  u32 out_slice_res_w_left_pixels = 0;
  u32 in_slice_res_w_left_pixels = 0;
  if (slice_res.w.left != 0) {
    int c_multiplier = m_chip_info.npu_num / channel;
    while (out_slice_res.w.left % c_multiplier != 0) {
      c_multiplier--;
    }
    u32 out_w_left_pixels = 0, new_c_multiplier = 0;
    if (updateIfExceedTLMem(m_chip_info.npu_num, c_multiplier, channel, out_info.tg_store.shape.w,
                            out_slice_res.w.left, m_kernel_info.default_stride_x, &new_c_multiplier,
                            &out_w_left_pixels)) {
      c_multiplier = new_c_multiplier;
      u32 in_w_left_pixels = out_w_left_pixels;
      out_slice_res.w.left -= out_w_left_pixels;
      out_slice_res.w.turn++;
      in_slice_res.w.left -= in_w_left_pixels;
      in_slice_res.w.turn++;
      slice_res.w.left = in_slice_res.w.left;
      slice_res.w.turn++;
      if (in_w_left_pixels >= m_kernel_info.size) {
        out_slice_res_w_left_pixels = out_w_left_pixels;
        in_slice_res_w_left_pixels = in_w_left_pixels;
      }
    }
    channelExtension(bk_ctx, channel, out_slice_res.h.slice, out_slice_res.w.left, c_multiplier,
                     m_kernel_info.pad[0], m_kernel_info.pad[1], m_kernel_info.pad[2],
                     m_kernel_info.pad[3], m_kernel_info.size, m_kernel_info.size,
                     m_kernel_info.default_stride_y, m_kernel_info.default_stride_x,
                     m_slice_info.io_fmt, &out_w_info);
  } else {
    out_w_info = out_info;
  }
  if (out_slice_res.h.left != 0) {
    u32 c_multiplier = m_chip_info.npu_num / channel;
    while (out_slice_res.h.left % c_multiplier != 0) {
      c_multiplier--;
    }
    u32 out_h_left_pixels = 0, new_c_multiplier = 0;
    if (updateIfExceedTLMem(m_chip_info.npu_num, c_multiplier, channel, out_info.tg_store.shape.h,
                            out_slice_res.h.left, m_kernel_info.default_stride_y, &new_c_multiplier,
                            &out_h_left_pixels)) {
      c_multiplier = new_c_multiplier;
      u32 in_h_left_pixels = out_h_left_pixels + vertical_pad_total;
      out_slice_res.h.left -= out_h_left_pixels;
      out_slice_res.h.turn++;
      in_slice_res.h.left -= in_h_left_pixels;
      in_slice_res.h.turn++;
      slice_res.h.left = in_slice_res.h.left;
      slice_res.h.turn++;
      if (in_h_left_pixels >= m_kernel_info.size) {
        out_slice_res_h_left_pixels = out_h_left_pixels;
        in_slice_res_h_left_pixels = in_h_left_pixels;
      }
    }
    channelExtension(bk_ctx, channel, out_slice_res.h.left, out_slice_res.w.slice, c_multiplier,
                     m_kernel_info.pad[0], m_kernel_info.pad[1], m_kernel_info.pad[2],
                     m_kernel_info.pad[3], m_kernel_info.size, m_kernel_info.size,
                     m_kernel_info.default_stride_y, m_kernel_info.default_stride_x,
                     m_slice_info.io_fmt, &out_h_info);
  } else {
    out_h_info = out_info;
  }
  if (out_slice_res.h.left != 0 && out_slice_res.w.left != 0) {
    int c_multiplier = m_chip_info.npu_num / channel;
    while (out_slice_res.h.left % c_multiplier != 0) {
      c_multiplier--;
    }
    while (out_slice_res.w.left % c_multiplier != 0) {
      c_multiplier--;
    }
    channelExtension(bk_ctx, channel, out_slice_res.h.left, out_slice_res.w.left, c_multiplier,
                     m_kernel_info.pad[0], m_kernel_info.pad[1], m_kernel_info.pad[2],
                     m_kernel_info.pad[3], m_kernel_info.size, m_kernel_info.size,
                     m_kernel_info.default_stride_y, m_kernel_info.default_stride_x,
                     m_slice_info.io_fmt, &out_wh_info);
  } else if (out_slice_res.h.left != 0 && out_slice_res.w.left == 0) {
    out_wh_info = out_h_info;
  } else if (out_slice_res.h.left == 0 && out_slice_res.w.left != 0) {
    out_wh_info = out_w_info;
  } else {
    out_wh_info = out_info;
  }

  TensorSliceInfo out_hlp_w_info, out_hlp_w_left_info, out_wlp_h_info, out_wlp_h_left_info,
      out_wlp_hlp_info;
  if (out_slice_res_h_left_pixels != 0) {
    updateTSIInfo(bk_ctx, 1, channel, in_slice_res_h_left_pixels, in_slice_res.w.slice, 1, channel,
                  out_slice_res_h_left_pixels, out_slice_res.w.slice, m_slice_info.io_fmt,
                  input[0].m_tg.fmt, &out_hlp_w_info);
    if (out_slice_res.w.left != 0) {
      updateTSIInfo(bk_ctx, 1, channel, in_slice_res_h_left_pixels, in_slice_res.w.left, 1, channel,
                    out_slice_res_h_left_pixels, out_slice_res.w.left, m_slice_info.io_fmt,
                    input[0].m_tg.fmt, &out_hlp_w_left_info);
    } else {
      out_hlp_w_left_info = out_hlp_w_info;
    }
  }
  if (out_slice_res_w_left_pixels != 0) {
    updateTSIInfo(bk_ctx, 1, channel, in_slice_res.h.slice, in_slice_res_w_left_pixels, 1, channel,
                  out_slice_res.h.slice, out_slice_res_w_left_pixels, m_slice_info.io_fmt,
                  input[0].m_tg.fmt, &out_wlp_h_info);
    if (out_slice_res.h.left != 0) {
      updateTSIInfo(bk_ctx, 1, channel, in_slice_res.h.left, in_slice_res_w_left_pixels, 1, channel,
                    out_slice_res.h.left, out_slice_res_w_left_pixels, m_slice_info.io_fmt,
                    input[0].m_tg.fmt, &out_wlp_h_left_info);
    } else {
      out_wlp_h_left_info = out_wlp_h_info;
    }
  }
  if (out_slice_res_h_left_pixels != 0 && out_slice_res_w_left_pixels != 0) {
    updateTSIInfo(bk_ctx, 1, channel, in_slice_res_h_left_pixels, in_slice_res_w_left_pixels, 1,
                  channel, out_slice_res_h_left_pixels, out_slice_res_w_left_pixels,
                  m_slice_info.io_fmt, input[0].m_tg.fmt, &out_wlp_hlp_info);
  }

  // Setup slice input/ output shapes and left shapes
  std::vector<bmk1880v2_tensor_tgmem_shape_t> s_in_vec, s_out_vec;
  for (size_t k = 0; k < input.size(); k++) {
    s_in_vec.push_back(
        {1, out_info.tl_load.shape.c, out_info.tl_load.shape.h, out_info.tl_load.shape.w});
  }
  for (size_t k = 0; k < output->size(); k++) {
    s_out_vec.push_back(
        {1, out_info.tl_store.shape.c, out_info.tl_store.shape.h, out_info.tl_store.shape.w});
  }

  // allocate tl shape and get input/ output indices.
  std::vector<u32> tl_in_idx, tl_out_idx;
  runSetup(ctx, bk_ctx, s_in_vec, s_out_vec, &tl_in_idx, &tl_out_idx, true);

  // Dummy check, can be turned off in official release
  if (tl_in_idx.size() != input.size()) {
    std::cerr << "Input tl size not match input image num " << tl_in_idx.size() << ", "
              << input.size() << std::endl;
  }
  if (tl_out_idx.size() != output->size()) {
    std::cerr << "Output tl size not match input image num " << tl_out_idx.size() << ", "
              << output->size() << std::endl;
  }
  // Dummy check end

  // Category tl shapes
  std::vector<std::pair<int, bmk1880v2_tensor_lmem_t *>> tl_in_shape_lmem_vec,
      tl_out_shape_lmem_vec;
  categoryIOTLShape(m_tl_vec, s_in_vec, s_out_vec, &tl_in_shape_lmem_vec, &tl_out_shape_lmem_vec);

  // Find and create input/ output fmt size pair.
  TLInfo tl_in_info, tl_out_info;
  getTLInfo(m_tl_vec, tl_in_idx, tl_out_idx, &tl_in_info, &tl_out_info);

  // Get device memory start offset
  BMAddrInfo bm_src_info, bm_dest_info;
  getBMAddrInfo(input, *output, m_kernel_info.pad[0], m_kernel_info.pad[2], &bm_src_info,
                &bm_dest_info);

  // Create tg block
  bmk1880v2_tensor_tgmem_t tg_in;
  tg_in.base_reg_index = 0;
  bmk1880v2_tensor_tgmem_t tg_out;
  tg_out.base_reg_index = 0;

  // Main for loop
  TensorSliceInfo *tsi = &out_info;
  for (u32 i = 0; i < slice_res.h.turn; i++) {
    // Re-assign head address to w.
    std::vector<u64> bm_src_addr_w = bm_src_info.addr_vec;
    std::vector<u64> bm_dest_addr_w = bm_dest_info.addr_vec;
    for (u32 j = 0; j < slice_res.w.turn; j++) {
      // Change H TL size to fit left shape in last turn
      // out_info       out_w_info          out_wlp_h_info
      // out_h_info     out_wh_info         out_wlp_h_left_info
      // out_hlp_w_info out_wlp_w_left_info out_wlp_hlp_info
      if (i == in_slice_res.h.turn - 1 && out_slice_res_h_left_pixels != 0) {
        if (j == 0) {
          tsi = &out_hlp_w_info;
          updateLMemSize(bk_ctx, m_chip_info.npu_num, m_slice_info.io_fmt, *tsi,
                         &tl_in_shape_lmem_vec, &tl_out_shape_lmem_vec, &m_tl_vec);
        } else if (j == in_slice_res.w.turn - 1 && out_slice_res_w_left_pixels != 0) {
          tsi = &out_wlp_hlp_info;
          updateLMemSize(bk_ctx, m_chip_info.npu_num, m_slice_info.io_fmt, *tsi,
                         &tl_in_shape_lmem_vec, &tl_out_shape_lmem_vec, &m_tl_vec);
        } else if ((j == in_slice_res.w.turn - 1 && out_slice_res_w_left_pixels == 0) ||
                   (j == in_slice_res.w.turn - 2 && out_slice_res_w_left_pixels != 0)) {
          tsi = &out_wlp_h_left_info;
          updateLMemSize(bk_ctx, m_chip_info.npu_num, m_slice_info.io_fmt, *tsi,
                         &tl_in_shape_lmem_vec, &tl_out_shape_lmem_vec, &m_tl_vec);
        }
      } else if ((i == in_slice_res.h.turn - 1 && out_slice_res_h_left_pixels == 0) ||
                 (i == in_slice_res.h.turn - 2 && out_slice_res_h_left_pixels != 0)) {
        if (j == 0) {
          tsi = &out_h_info;
          updateLMemSize(bk_ctx, m_chip_info.npu_num, m_slice_info.io_fmt, *tsi,
                         &tl_in_shape_lmem_vec, &tl_out_shape_lmem_vec, &m_tl_vec);
        } else if (j == in_slice_res.w.turn - 1 && out_slice_res_w_left_pixels != 0) {
          tsi = &out_wlp_h_left_info;
          updateLMemSize(bk_ctx, m_chip_info.npu_num, m_slice_info.io_fmt, *tsi,
                         &tl_in_shape_lmem_vec, &tl_out_shape_lmem_vec, &m_tl_vec);
        } else if ((j == in_slice_res.w.turn - 1 && out_slice_res_w_left_pixels == 0) ||
                   (j == in_slice_res.w.turn - 2 && out_slice_res_w_left_pixels != 0)) {
          tsi = &out_wh_info;
          updateLMemSize(bk_ctx, m_chip_info.npu_num, m_slice_info.io_fmt, *tsi,
                         &tl_in_shape_lmem_vec, &tl_out_shape_lmem_vec, &m_tl_vec);
        }
      } else {
        if (j == 0) {
          tsi = &out_info;
          updateLMemSize(bk_ctx, m_chip_info.npu_num, m_slice_info.io_fmt, *tsi,
                         &tl_in_shape_lmem_vec, &tl_out_shape_lmem_vec, &m_tl_vec);
        } else if (j == in_slice_res.w.turn - 1 && out_slice_res_w_left_pixels != 0) {
          tsi = &out_wlp_h_info;
          updateLMemSize(bk_ctx, m_chip_info.npu_num, m_slice_info.io_fmt, *tsi,
                         &tl_in_shape_lmem_vec, &tl_out_shape_lmem_vec, &m_tl_vec);
        } else if ((j == in_slice_res.w.turn - 1 && out_slice_res_w_left_pixels == 0) ||
                   (j == in_slice_res.w.turn - 2 && out_slice_res_w_left_pixels != 0)) {
          tsi = &out_w_info;
          updateLMemSize(bk_ctx, m_chip_info.npu_num, m_slice_info.io_fmt, *tsi,
                         &tl_in_shape_lmem_vec, &tl_out_shape_lmem_vec, &m_tl_vec);
        }
      }

      // tg2tl
      for (size_t k = 0; k < tl_in_info.lmem_vec.size(); k++) {
        auto &tl_in = tl_in_info.lmem_vec;
        tg_in.start_address = bm_src_addr_w[k];
        tg_in.shape = tsi->tg_load.shape;
        tg_in.stride = tsi->tg_load.stride;
        tg_in.fmt = bm_src_info.fns_vec[k].getFmt();

        // clang-format off
        IVE_DEBUG("[%lu] In\n"
                  " tg start address %lu\n"
                  " tg shape %d %d %d %d\n"
                  " tg stride %d %d %d\n"
                  " tl shape %d %d %d %d\n"
                  " tl stride %u %u %u %u\n",
                  k,
                  tg_in.start_address,
                  tg_in.shape.n, tg_in.shape.c, tg_in.shape.h, tg_in.shape.w,
                  tg_in.stride.n, tg_in.stride.c, tg_in.stride.h,
                  tl_in[k]->shape.n, tl_in[k]->shape.c, tl_in[k]->shape.h, tl_in[k]->shape.w,
                  tl_in[k]->stride.n, tl_in[k]->stride.c, tl_in[k]->stride.h, tl_in[k]->stride.w);
        // clang-format on

        bmk1880v2_tdma_tg2l_tensor_copy_param_t p_copy_in;
        p_copy_in.src = &tg_in;
        p_copy_in.dst = tl_in[k];
        bmk1880v2_tdma_g2l_bf16_tensor_copy(bk_ctx, &p_copy_in);

        // Change src head addr
        bm_src_addr_w[k] += 1 * in_slice_res.w.skip * bm_src_info.fns_vec[k].getSize();
      }

      operation(ctx, bk_ctx, 0);

      // tl2tg
      for (size_t k = 0; k < tl_out_info.lmem_vec.size(); k++) {
        tg_out.start_address = bm_dest_addr_w[k];
        tg_out.fmt = bm_dest_info.fns_vec[k].getFmt();
        tg_out.shape = tsi->tg_store.shape;
        // Note: Channel extension does not support h padding.
        // tg_out.shape.h = tl_out[k]->shape.h - (m_kernel_info.pad[2] + m_kernel_info.pad[3]);
        tg_out.shape.w = tg_out.shape.w - (m_kernel_info.pad[0] + m_kernel_info.pad[1]);
        tg_out.stride = tsi->tg_store.stride;
        auto &tl_out = tl_out_info.lmem_vec;
        bmk1880v2_tensor_lmem_t out_shape;
        // printf("st addr%d, tg st addr %lu\n", tl_out[k]->start_address, bm_dest_addr_w[k]);
        out_shape.start_address =
            tl_out[k]->start_address + (m_kernel_info.pad[0] * tl_out_info.fns_vec[k].getSize());
        out_shape.fmt = tl_out[k]->fmt;
        out_shape.cmprs_fmt = tl_out[k]->cmprs_fmt;
        out_shape.shape = tl_out[k]->shape;
        out_shape.stride = tl_out[k]->stride;
        out_shape.shape.w = tg_out.shape.w;

        // clang-format off
        IVE_DEBUG("[%lu] Out\n"
                  " tg start address %lu\n"
                  " tg shape %d %d %d %d\n"
                  " tg stride %d %d %d\n"
                  " tl shape %d %d %d %d\n"
                  " tl stride %u %u %u %u\n",
                  k,
                  tg_out.start_address,
                  tg_out.shape.n, tg_out.shape.c, tg_out.shape.h, tg_out.shape.w,
                  tg_out.stride.n, tg_out.stride.c, tg_out.stride.h,
                  tl_out[k]->shape.n, tl_out[k]->shape.c, tl_out[k]->shape.h, tl_out[k]->shape.w,
                  tl_out[k]->stride.n, tl_out[k]->stride.c, tl_out[k]->stride.h, tl_out[k]->stride.w);
        // clang-format on

        bmk1880v2_tdma_l2tg_tensor_copy_param_t p_copy_out;
        p_copy_out.src = &out_shape;
        p_copy_out.dst = &tg_out;
        bmk1880v2_tdma_l2g_bf16_tensor_copy(bk_ctx, &p_copy_out);

        // Change dest head addr
        bm_dest_addr_w[k] += 1 * out_slice_res.w.skip * bm_dest_info.fns_vec[k].getSize();
      }
    }
    // Change src/ dest head addr
    for (size_t k = 0; k < bm_src_info.addr_vec.size(); k++) {
      u32 jump_val = 0;
      if (in_slice_res_h_left_pixels != 0) {
        if (i == in_slice_res.h.turn - 1) {
          jump_val = in_slice_res_h_left_pixels;
        } else if (i == in_slice_res.h.turn - 2 && in_slice_res.h.left != 0) {
          jump_val = in_slice_res.h.left;
        } else {
          jump_val = in_slice_res.h.skip;
        }
      } else {
        jump_val = (i == in_slice_res.h.turn - 1 && in_slice_res.h.left != 0) ? in_slice_res.h.left
                                                                              : in_slice_res.h.skip;
      }
      bm_src_info.addr_vec[k] += 1 * input[k].m_tg.stride.h * jump_val;
    }
    for (size_t k = 0; k < bm_dest_info.addr_vec.size(); k++) {
      u32 jump_val = 0;
      if (out_slice_res_h_left_pixels != 0) {
        if (i == out_slice_res.h.turn - 1) {
          jump_val = out_slice_res_h_left_pixels + vertical_pad_total;
        } else if (i == out_slice_res.h.turn - 2 && out_slice_res.h.left != 0) {
          jump_val = out_slice_res.h.left;
        } else {
          jump_val = out_slice_res.h.skip;
        }
      } else {
        jump_val = (i == out_slice_res.h.turn - 1 && out_slice_res.h.left != 0)
                       ? out_slice_res.h.left + vertical_pad_total
                       : out_slice_res.h.skip;
      }
      bm_dest_info.addr_vec[k] += 1 * (*output)[k].m_tg.stride.h * jump_val;
    }
  }
  IVE_DEBUG("In slice info:\n");
  IVE_DEBUG("{ h_slice, h_turn, h_skip, h_left} = { %d, %d, %d, %d}\n", in_slice_res.h.slice,
            in_slice_res.h.turn, in_slice_res.h.skip, in_slice_res.h.left);
  IVE_DEBUG("{ w_slice, w_turn, w_skip, w_left} = { %d, %d, %d, %d}\n", in_slice_res.w.slice,
            in_slice_res.w.turn, in_slice_res.w.skip, in_slice_res.w.left);
  IVE_DEBUG("Out slice info:\n");
  IVE_DEBUG("{ h_slice, h_turn, h_skip, h_left} = { %d, %d, %d, %d}\n", out_slice_res.h.slice,
            out_slice_res.h.turn, out_slice_res.h.skip, out_slice_res.h.left);
  IVE_DEBUG("{ w_slice, w_turn, w_skip, w_left} = { %d, %d, %d, %d}\n", out_slice_res.w.slice,
            out_slice_res.w.turn, out_slice_res.w.skip, out_slice_res.w.left);

  // Dummy gaurd for buffer overflow
  ret |= checkIsBufferOverflow(input, *output, bm_src_info, bm_dest_info, m_kernel_info.pad[0],
                               m_kernel_info.pad[2], false);
  if (ret == BM_SUCCESS) {
    bmruntime_bmkernel_submit(*ctx);
  }

  freeTLMems(bk_ctx);
  freeChildTGMem(ctx);
  return BM_SUCCESS;
}

int IveCore::runNoKernel(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx, std::vector<CviImg> &input,
                         std::vector<CviImg> *output, bool enable_min_max) {
  // Only supports kernel size = 1. NoKernel means kernel size = 1. You still can use depthwise
  // conv + qdm as u8 div.
  if (m_kernel_info.size != 1) {
    return BM_ERR_FAILURE;
  }
  u32 total_size = input[0].m_tg.stride.n / getFmtSize(input[0].m_tg.fmt);
  if (total_size % 16) {
    std::cerr << "Image size " << total_size << " is not 16 aligned." << std::endl;
    return BM_ERR_FAILURE;
  }
  bmk1880v2_tensor_lmem_shape_t tl_table_s;
  u64 table_sz = bf16_lut_tbl_bytesize(bk_ctx, &tl_table_s, FMT_U8);
  m_table_per_channel_size = table_sz / m_chip_info.npu_num;  // 32 * 8 for bm1880v2
  // Calculating slice
  // Calculate fixed kernel size
  u32 kernel_sz = (m_kernel_info.nums_of_kernel * m_kernel_info.size * m_kernel_info.size +
                   MULTIPLIER_ONLY_PACKED_DATA_SIZE * m_kernel_info.use_multiplier);
  // Find max available mem for one tl.
  int64_t result = m_chip_info.lmem_size -
                   (int64_t)(kernel_sz + m_table_per_channel_size * m_slice_info.nums_of_table) -
                   (int64_t)m_slice_info.fix_lmem_size;
  size_t max_hxw =
      std::floor(result / ((m_slice_info.nums_of_tl - m_slice_info.ping_pong_share_tl) *
                               m_slice_info.ping_pong_size +
                           m_slice_info.ping_pong_share_tl));
  u32 idiv_32 = (u32)(total_size / 32);
  uint32_t div = max_hxw;
  // Find div value that idiv % div == 0 while div < max_hxw
  while (idiv_32 % div != 0) {
    u32 val = std::ceil(float(idiv_32) / div);
    div = std::floor(float(idiv_32) / val);
  }
  // Make w 16 align.
  u32 div_16 = div / 16;
  div = div_16 * 16;
  // FIXME: We assumed that h never exceeds 1024.
  bmk1880v2_tensor_tgmem_shape_t shape = {1, 32, div_16, 16};
  size_t loop_turn = (total_size / (32 * div)) / m_slice_info.ping_pong_size;
  std::vector<bmk1880v2_tensor_tgmem_shape_t> s_in_vec, s_out_vec;
  for (size_t k = 0; k < input.size(); k++) {
    s_in_vec.push_back({shape.n, shape.c, shape.h, shape.w});
  }
  for (size_t k = 0; k < output->size(); k++) {
    s_out_vec.push_back({shape.n, shape.c, shape.h, shape.w});
  }
  // Check if any pixel left.
  size_t left_pixels = total_size - ((loop_turn * (32 * div)) * m_slice_info.ping_pong_size);
  IVE_DEBUG("Total size %u\n", total_size);
  IVE_DEBUG("turn %lu left %lu\n", loop_turn, left_pixels);
  IVE_DEBUG("%u %u %u %u\n", shape.n, shape.c, shape.h, shape.w);

  // allocate tl shape and get input/ output indices.
  std::vector<u32> tl_in_idx, tl_out_idx;
  runSetup(ctx, bk_ctx, s_in_vec, s_out_vec, &tl_in_idx, &tl_out_idx, false);

  // Find and create input/ output fmt size pair.
  TLInfo tl_in_info, tl_out_info;
  getTLInfo(m_tl_vec, tl_in_idx, tl_out_idx, &tl_in_info, &tl_out_info);

  // Get device memory start offset
  BMAddrInfo bm_src_info, bm_dest_info;
  getBMAddrInfo(input, *output, m_kernel_info.pad[0], m_kernel_info.pad[2], &bm_src_info,
                &bm_dest_info);
  // Get reshaped stride
  std::vector<bmk1880v2_tensor_tgmem_stride_t> input_stride_vec, output_stride_vec;
  for (size_t i = 0; i < bm_src_info.addr_vec.size(); i++) {
    input_stride_vec.push_back(
        bmk1880v2_bf16_tensor_tgmem_default_stride(shape, input[0].m_tg.fmt));
  }
  for (size_t i = 0; i < bm_src_info.addr_vec.size(); i++) {
    output_stride_vec.push_back(
        bmk1880v2_bf16_tensor_tgmem_default_stride(shape, (*output)[0].m_tg.fmt));
  }

  // Create tg block
  bmk1880v2_tensor_tgmem_t tg_in;
  tg_in.base_reg_index = 0;
  bmk1880v2_tensor_tgmem_t tg_out;
  tg_out.base_reg_index = 0;

  size_t jump_src = shape.n * shape.c * shape.h * shape.w;
  size_t jump_dst = jump_src;
  for (size_t i = 0; i < loop_turn; i++) {
    for (size_t pp = 0; pp < m_slice_info.ping_pong_size; pp++) {
      u32 tl_idx = tl_in_info.lmem_vec.size() / m_slice_info.ping_pong_size;
      u32 pp_skip = pp * tl_idx;
      for (size_t k = 0; k < tl_idx; k++) {
        tg_in.start_address = bm_src_info.addr_vec[k];
        tg_in.shape.n = tl_in_info.lmem_vec[k + pp_skip]->shape.n;
        tg_in.shape.c = tl_in_info.lmem_vec[k + pp_skip]->shape.c;
        tg_in.shape.h = tl_in_info.lmem_vec[k + pp_skip]->shape.h;
        tg_in.shape.w = tl_in_info.lmem_vec[k + pp_skip]->shape.w;
        tg_in.fmt = bm_src_info.fns_vec[k].getFmt();
        tg_in.stride = input_stride_vec[k];
        bmk1880v2_tdma_tg2l_tensor_copy_param_t p_copy_in;
        p_copy_in.src = &tg_in;
        p_copy_in.dst = tl_in_info.lmem_vec[k + pp_skip];
        bmk1880v2_tdma_g2l_bf16_tensor_copy(bk_ctx, &p_copy_in);
        // Change src head addr
        bm_src_info.addr_vec[k] += 1 * jump_src * bm_src_info.fns_vec[k].getSize();
      }
    }

    for (size_t pp = 0; pp < m_slice_info.ping_pong_size; pp++) {
      operation(ctx, bk_ctx, pp);
    }

    // tl2tg
    for (size_t pp = 0; pp < m_slice_info.ping_pong_size; pp++) {
      u32 tl_idx = tl_out_info.lmem_vec.size() / m_slice_info.ping_pong_size;
      u32 pp_skip = pp * tl_idx;
      for (size_t k = 0; k < tl_idx; k++) {
        tg_out.start_address = bm_dest_info.addr_vec[k];
        tg_out.shape.n = tl_out_info.lmem_vec[k + pp_skip]->shape.n;
        tg_out.shape.c = tl_out_info.lmem_vec[k + pp_skip]->shape.c;
        tg_out.shape.h = tl_out_info.lmem_vec[k + pp_skip]->shape.h;
        tg_out.shape.w = tl_out_info.lmem_vec[k + pp_skip]->shape.w;
        tg_out.fmt = bm_dest_info.fns_vec[k].getFmt();
        tg_out.stride = output_stride_vec[k];
        bmk1880v2_tensor_lmem_t out_shape;
        bmk1880v2_tdma_l2tg_tensor_copy_param_t p_copy_out;
        p_copy_out.src = tl_out_info.lmem_vec[k + pp_skip];
        p_copy_out.dst = &tg_out;
        bmk1880v2_tdma_l2g_bf16_tensor_copy(bk_ctx, &p_copy_out);

        // Change dest head addr
        bm_dest_info.addr_vec[k] += 1 * jump_dst * bm_dest_info.fns_vec[k].getSize();
      }
    }
  }
  if (left_pixels != 0) {
    bmk1880v2_tensor_tgmem_shape_t left_shape = {0, 0, 0, 0};
    u32 div = 32;
    while (left_pixels % div != 0) {
      u32 val = std::ceil(float(left_pixels) / div);
      div = std::floor(float(left_pixels) / val);
    }
    u32 hw = left_pixels / div;
    // FIXME: Again, we assumed that h and w may not exceed 1024.
    u32 w_val = 1024;
    while (hw % w_val != 0) {
      u32 val = std::ceil(float(hw) / w_val);
      w_val = std::floor(float(hw) / val);
    }
    left_shape.n = 1;
    left_shape.c = div;
    left_shape.h = hw / w_val;
    left_shape.w = w_val;
    IVE_DEBUG("%u %u %u %u\n", left_shape.n, left_shape.c, left_shape.h, left_shape.w);

    for (size_t i = 0; i < input_stride_vec.size(); i++) {
      input_stride_vec[i] =
          bmk1880v2_bf16_tensor_tgmem_default_stride(left_shape, input[0].m_tg.fmt);
    }
    for (size_t i = 0; i < output_stride_vec.size(); i++) {
      output_stride_vec[i] =
          bmk1880v2_bf16_tensor_tgmem_default_stride(left_shape, (*output)[0].m_tg.fmt);
    }
    // Category tl shapes
    for (size_t i = 0; i < m_tl_vec.size(); i++) {
      auto *lmem = m_tl_vec[i];
      bool skip = false;
      for (size_t k = 0; k < s_in_vec.size(); k++) {
        if (tgTLShapeCompare(lmem->shape, s_in_vec[k])) {
          lmem->shape.n = left_shape.n;
          lmem->shape.c = left_shape.c;
          lmem->shape.h = left_shape.h;
          lmem->shape.w = left_shape.w;
          lmem->stride =
              bmk1880v2_bf16_tensor_lmem_default_stride(bk_ctx, lmem->shape, 1, lmem->fmt);
          skip = true;
          break;
        }
      }
      if (skip) {
        continue;
      }
      for (size_t k = 0; k < s_out_vec.size(); k++) {
        if (tgTLShapeCompare(lmem->shape, s_out_vec[k])) {
          lmem->shape.n = left_shape.n;
          lmem->shape.c = left_shape.c;
          lmem->shape.h = left_shape.h;
          lmem->shape.w = left_shape.w;
          lmem->stride =
              bmk1880v2_bf16_tensor_lmem_default_stride(bk_ctx, lmem->shape, 1, lmem->fmt);
          break;
        }
      }
    }
    size_t jump_src = left_shape.n * left_shape.c * left_shape.h * left_shape.w;
    size_t jump_dst = jump_src;
    u32 tl_idx = tl_in_info.lmem_vec.size() / m_slice_info.ping_pong_size;
    for (size_t k = 0; k < tl_idx; k++) {
      tg_in.start_address = bm_src_info.addr_vec[k];
      tg_in.shape.n = tl_in_info.lmem_vec[k]->shape.n;
      tg_in.shape.c = tl_in_info.lmem_vec[k]->shape.c;
      tg_in.shape.h = tl_in_info.lmem_vec[k]->shape.h;
      tg_in.shape.w = tl_in_info.lmem_vec[k]->shape.w;
      tg_in.fmt = bm_src_info.fns_vec[k].getFmt();
      tg_in.stride = input_stride_vec[k];
      bmk1880v2_tdma_tg2l_tensor_copy_param_t p_copy_in;
      p_copy_in.src = &tg_in;
      p_copy_in.dst = tl_in_info.lmem_vec[k];
      bmk1880v2_tdma_g2l_bf16_tensor_copy(bk_ctx, &p_copy_in);

      // Change src head addr
      bm_src_info.addr_vec[k] += 1 * jump_src * bm_src_info.fns_vec[k].getSize();
    }

    operation(ctx, bk_ctx, 0);

    // tl2tg
    tl_idx = tl_out_info.lmem_vec.size() / m_slice_info.ping_pong_size;
    for (size_t k = 0; k < tl_idx; k++) {
      tg_out.start_address = bm_dest_info.addr_vec[k];
      tg_out.fmt = bm_dest_info.fns_vec[k].getFmt();
      tg_out.shape.n = tl_out_info.lmem_vec[k]->shape.n;
      tg_out.shape.c = tl_out_info.lmem_vec[k]->shape.c;
      tg_out.shape.h = tl_out_info.lmem_vec[k]->shape.h;
      tg_out.shape.w = tl_out_info.lmem_vec[k]->shape.w;
      tg_out.stride = output_stride_vec[k];
      bmk1880v2_tensor_lmem_t out_shape;
      bmk1880v2_tdma_l2tg_tensor_copy_param_t p_copy_out;
      p_copy_out.src = tl_out_info.lmem_vec[k];
      p_copy_out.dst = &tg_out;
      bmk1880v2_tdma_l2g_bf16_tensor_copy(bk_ctx, &p_copy_out);

      // Change dest head addr
      bm_dest_info.addr_vec[k] += 1 * jump_dst * bm_dest_info.fns_vec[k].getSize();
    }
  }
  int ret = BM_SUCCESS;
  ret |= checkIsBufferOverflow(input, *output, bm_src_info, bm_dest_info, m_kernel_info.pad[0],
                               m_kernel_info.pad[2], true);
  if (ret == BM_SUCCESS) {
    if (m_write_cdbuf) {
      u32 len;
      u8 *buf = bmk1880v2_acquire_cmdbuf(bk_ctx, &len);
      printf("Cmdbuf length %u\n", len);
      FILE *pFile;
      pFile = fopen("cmdbuf.bin", "wb");
      fwrite(buf, sizeof(char), len, pFile);
      fclose(pFile);
      uint16_t seq_no;
      bmerr_t ret = bm_send_cmdbuf(*ctx, buf, (size_t)len, &seq_no);
      bmk1880v2_reset(bk_ctx);
    } else {
      bmruntime_bmkernel_submit(*ctx);
    }
  }

  freeTLMems(bk_ctx);
  freeChildTGMem(ctx);
  return BM_SUCCESS;
}