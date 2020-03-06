#include "core.hpp"

#include <cmath>
#include <iostream>
#include "debug.hpp"

IveCore::IveCore() {
  auto chip_info = bmk1880v2_chip_info();
  m_chip_info.eu_num = chip_info.eu_num;
  m_chip_info.lmem_bank_size = chip_info.lmem_bank_size;
  m_chip_info.lmem_banks = chip_info.lmem_banks;
  m_chip_info.lmem_size = chip_info.lmem_size;
  m_chip_info.npu_num = chip_info.npu_num;
  m_chip_info.version = chip_info.version;
}

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
    unit->left = result;
    unit->turn++;
  } else if (result < 0 && unit->slice > std::abs(result)) {
    unit->left = unit->slice + result;
  } else {
    unit->left = 0;
  }
}

int IveCore::getSlice(const u32 nums_of_lmem, const u32 nums_of_table, const u32 fixed_lmem_size,
                      const u32 n, const u32 c, const u32 h, const u32 w, const u32 table_size,
                      const kernelInfo kernel_info, sliceUnit *unit_h, sliceUnit *unit_w) {
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
  while (h_tmp_slice < kernel_info.size) {
    w_length = w / w_num;
    h_tmp_slice = available_lmem_per_tl / w_length;
    w_num++;
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
                                 std::vector<CviImg> &input, std::vector<CviImg> *output) {
  // FIXME: Support later
  if (m_slice_info.ping_pong_size != 1) {
    std::cerr << "Currently runSingleSizeKernel does not support ping pong." << std::endl;
    m_slice_info.ping_pong_size = 1;
  }
  u32 batch = input[0].m_tg.shape.n;
  u32 channel = input[0].m_tg.shape.c;
  u32 height = input[0].m_tg.shape.h;
  u32 width = input[0].m_tg.shape.w;
  // FIXME: Move to constructor if possible.
  bmk1880v2_tensor_lmem_shape_t tl_table_s;
  u64 result = bf16_lut_tbl_bytesize(bk_ctx, &tl_table_s, FMT_U8);
  m_table_per_channel_size = result / m_chip_info.npu_num;  // 32 * 8 for bm1880v2
  SliceRes slice_res;
  int ret = getSlice(m_slice_info.nums_of_tl, m_slice_info.nums_of_table,
                     m_slice_info.fix_lmem_size, batch, channel, height, width,
                     m_table_per_channel_size, m_kernel_info, &slice_res.h, &slice_res.w);
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
  runSetup(ctx, bk_ctx, s_in_vec, s_out_vec, &tl_in_idx, &tl_out_idx);

  // Dummy check, can be turned off in official release
  std::vector<bmk1880v2_tensor_lmem_t *> tl_in, tl_out;
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
  for (size_t i = 0; i < m_tl_vec.size(); i++) {
    auto *lmem = m_tl_vec[i];
    bool skip = false;
    for (size_t k = 0; k < s_in_vec.size(); k++) {
      if (tgTLShapeCompare(lmem->shape, s_in_vec[k])) {
        tl_in_shape_lmem_vec.push_back({k, lmem});
        skip = true;
        break;
      }
    }
    if (skip) {
      continue;
    }
    for (size_t k = 0; k < s_out_vec.size(); k++) {
      if (tgTLShapeCompare(lmem->shape, s_out_vec[k])) {
        tl_out_shape_lmem_vec.push_back({k, lmem});
        break;
      }
    }
  }

  // Find and create input/ output fmt size pair.
  std::vector<FmtPair> fmt_input_vec, fmt_output_vec;
  for (size_t i = 0; i < tl_in_idx.size(); i++) {
    auto *lmem = m_tl_vec[tl_in_idx[i]];
    tl_in.emplace_back(lmem);
    FmtPair fp;
    fp.setTGFmt(input[i].m_tg.fmt);
    fp.setTLFmt(lmem->fmt);
    fmt_input_vec.push_back(fp);
  }
  for (size_t i = 0; i < tl_out_idx.size(); i++) {
    auto *lmem = m_tl_vec[tl_out_idx[i]];
    tl_out.emplace_back(lmem);
    FmtPair fp;
    fp.setTGFmt((*output)[i].m_tg.fmt);
    fp.setTLFmt(lmem->fmt);
    fmt_output_vec.push_back(fp);
  }

  // Create tg block
  bmk1880v2_tensor_tgmem_t tg_in;
  tg_in.base_reg_index = 0;
  bmk1880v2_tensor_tgmem_t tg_out;
  tg_out.base_reg_index = 0;

  // Get device memory start offset
  std::vector<u64> bm_src_addr, bm_dest_addr;
  for (size_t k = 0; k < input.size(); k++) {
    u64 bm_start_addr = input[k].GetPAddr();
    bm_src_addr.push_back(bm_start_addr);
  }
  for (size_t k = 0; k < output->size(); k++) {
    u64 bm_des_addr = (*output)[k].GetPAddr();
    bm_dest_addr.push_back(bm_des_addr + ((*output)[k].m_tg.stride.h * m_kernel_info.pad[2]) +
                           (m_kernel_info.pad[0] * fmt_output_vec[k].getTGFmtSize()));
  }

  // Main for loop
  for (u32 i = 0; i < slice_res.h.turn; i++) {
    // Re-assign head address to w.
    std::vector<u64> bm_src_addr_w = bm_src_addr;
    std::vector<u64> bm_dest_addr_w = bm_dest_addr;
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
      for (size_t k = 0; k < tl_in.size(); k++) {
        tg_in.start_address = bm_src_addr_w[k];
        tg_in.shape.n = tl_in[k]->shape.n;
        tg_in.shape.c = tl_in[k]->shape.c;
        tg_in.shape.h = tl_in[k]->shape.h;
        tg_in.shape.w = tl_in[k]->shape.w;
        tg_in.fmt = fmt_input_vec[k].getTGFmt();
        tg_in.stride = input[k].m_tg.stride;
        bmk1880v2_tdma_tg2l_tensor_copy_param_t p_copy_in;
        p_copy_in.src = &tg_in;
        p_copy_in.dst = tl_in[k];
        bmk1880v2_tdma_g2l_bf16_tensor_copy(bk_ctx, &p_copy_in);

        // Change src head addr
        bm_src_addr_w[k] += 1 * in_slice_res.w.skip * fmt_input_vec[k].getTGFmtSize();
      }

      operation(ctx, bk_ctx, 0);

      // tl2tg
      for (size_t k = 0; k < tl_out.size(); k++) {
        tg_out.start_address = bm_dest_addr_w[k];
        tg_out.fmt = fmt_output_vec[k].getTGFmt();
        tg_out.shape.n = tl_out[k]->shape.n;
        tg_out.shape.c = tl_out[k]->shape.c;
        tg_out.shape.h = tl_out[k]->shape.h - (m_kernel_info.pad[2] + m_kernel_info.pad[3]);
        tg_out.shape.w = tl_out[k]->shape.w - (m_kernel_info.pad[0] + m_kernel_info.pad[1]);
        tg_out.stride = (*output)[k].m_tg.stride;
        bmk1880v2_tensor_lmem_t out_shape;
        // printf("st addr%d, tg st addr %lu\n", tl_out[k]->start_address, bm_dest_addr_w[k]);
        out_shape.start_address = tl_out[k]->start_address +
                                  (1 * tl_out[k]->stride.h * m_kernel_info.pad[2]) +
                                  (m_kernel_info.pad[0] * fmt_output_vec[k].getTLFmtSize());
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
        bm_dest_addr_w[k] += 1 * out_slice_res.w.skip * fmt_output_vec[k].getTGFmtSize();
      }
    }
    // Change src/ dest head addr
    for (size_t k = 0; k < tl_in.size(); k++) {
      auto jump_val = (i == in_slice_res.h.turn - 1 && in_slice_res.h.left != 0)
                          ? in_slice_res.h.left
                          : in_slice_res.h.skip;
      bm_src_addr[k] += 1 * input[k].m_tg.stride.h * jump_val;
    }
    for (size_t k = 0; k < tl_out.size(); k++) {
      auto jump_val = (i == out_slice_res.h.turn - 1 && out_slice_res.h.left != 0)
                          ? out_slice_res.h.left
                          : out_slice_res.h.skip;
      bm_dest_addr[k] += 1 * (*output)[k].m_tg.stride.h * jump_val;
    }
  }

  // Dummy gaurd for buffer overflow
  for (size_t k = 0; k < input.size(); k++) {
    u64 bm_start_addr = input[k].GetPAddr();
    u64 jumped_value = bm_src_addr[k] - bm_start_addr;
    u32 total_addr = input[k].m_tg.stride.n;
    if (jumped_value != total_addr) {
      printf("Error! Input %lu jumped value not align to image size %lu, original %u\n", k,
             jumped_value, total_addr);
      ret = BM_ERR_FAILURE;
    }
  }
  for (size_t k = 0; k < output->size(); k++) {
    u64 bm_des_addr = (*output)[k].GetPAddr();
    u64 jumped_value = bm_dest_addr[k] - bm_des_addr;
    u32 total_addr =
        ((*output)[k].m_tg.stride.n + (((*output)[k].m_tg.stride.h * m_kernel_info.pad[2]) +
                                       (m_kernel_info.pad[0] * fmt_output_vec[k].getTGFmtSize())));
    if (jumped_value != total_addr) {
      printf("Error! Output %lu jumped value not align to image size %lu, original %u\n", k,
             jumped_value, total_addr);
      ret = BM_ERR_FAILURE;
    }
  }
  IVE_DEBUG("Slice info:\n");
  IVE_DEBUG("{ h_slice, h_turn, h_skip, h_left} = { %d, %d, %d, %d}\n", m_slice_info.h.slice,
            m_slice_info.h.turn, m_slice_info.h.skip, m_slice_info.h.left);
  IVE_DEBUG("{ w_slice, w_turn, w_skip, w_left} = { %d, %d, %d, %d}\n", m_slice_info.w.slice,
            m_slice_info.w.turn, m_slice_info.w.skip, m_slice_info.w.left);
  if (ret == BM_SUCCESS) {
    bmruntime_bmkernel_submit(*ctx);
  }
  freeTLMems(bk_ctx);
  freeChildTGMem(ctx);
  return ret;
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
  runSetup(ctx, bk_ctx, s_in_vec, s_out_vec, &tl_in_idx, &tl_out_idx);

  // Find and create input/ output fmt size pair.
  std::vector<bmk1880v2_tensor_lmem_t *> tl_in, tl_out;
  std::vector<FmtPair> fmt_input_vec, fmt_output_vec;
  std::vector<bmk1880v2_tensor_tgmem_stride_t> input_stride_vec, output_stride_vec;
  u32 sz_mod = tl_in_idx.size() / m_slice_info.ping_pong_size;
  for (size_t i = 0; i < tl_in_idx.size(); i++) {
    auto *lmem = m_tl_vec[tl_in_idx[i]];
    tl_in.emplace_back(lmem);
    FmtPair fp;
    fp.setTGFmt(input[i % sz_mod].m_tg.fmt);
    fp.setTLFmt(lmem->fmt);
    fmt_input_vec.push_back(fp);
    input_stride_vec.push_back(
        bmk1880v2_bf16_tensor_tgmem_default_stride(shape, input[0].m_tg.fmt));
  }
  sz_mod = tl_out_idx.size() / m_slice_info.ping_pong_size;
  for (size_t i = 0; i < tl_out_idx.size(); i++) {
    auto *lmem = m_tl_vec[tl_out_idx[i]];
    tl_out.emplace_back(lmem);
    FmtPair fp;
    fp.setTGFmt((*output)[i % sz_mod].m_tg.fmt);
    fp.setTLFmt(lmem->fmt);
    fmt_output_vec.push_back(fp);
    output_stride_vec.push_back(
        bmk1880v2_bf16_tensor_tgmem_default_stride(shape, (*output)[0].m_tg.fmt));
  }

  // Create tg block
  bmk1880v2_tensor_tgmem_t tg_in;
  tg_in.base_reg_index = 0;
  bmk1880v2_tensor_tgmem_t tg_out;
  tg_out.base_reg_index = 0;

  // Get device memory start offset
  std::vector<u64> bm_src_addr, bm_dest_addr;
  for (size_t k = 0; k < input.size(); k++) {
    u64 bm_start_addr = input[k].GetPAddr();
    bm_src_addr.push_back(bm_start_addr);
  }
  for (size_t k = 0; k < output->size(); k++) {
    u64 bm_des_addr = (*output)[k].GetPAddr();
    bm_dest_addr.push_back(bm_des_addr + ((*output)[k].m_tg.stride.h * m_kernel_info.pad[2]) +
                           (m_kernel_info.pad[0] * fmt_output_vec[k].getTGFmtSize()));
  }
  // FIXME: Currently only supports input image size = output image.
  size_t jump_src = shape.n * shape.c * shape.h * shape.w;
  size_t jump_dst = jump_src;
  for (size_t i = 0; i < loop_turn; i++) {
    for (size_t pp = 0; pp < m_slice_info.ping_pong_size; pp++) {
      u32 tl_idx = tl_in.size() / m_slice_info.ping_pong_size;
      u32 pp_skip = pp * tl_idx;
      for (size_t k = 0; k < tl_idx; k++) {
        tg_in.start_address = bm_src_addr[k];
        tg_in.shape.n = tl_in[k + pp_skip]->shape.n;
        tg_in.shape.c = tl_in[k + pp_skip]->shape.c;
        tg_in.shape.h = tl_in[k + pp_skip]->shape.h;
        tg_in.shape.w = tl_in[k + pp_skip]->shape.w;
        tg_in.fmt = fmt_input_vec[k + pp_skip].getTGFmt();
        tg_in.stride = input_stride_vec[k + pp_skip];
        bmk1880v2_tdma_tg2l_tensor_copy_param_t p_copy_in;
        p_copy_in.src = &tg_in;
        p_copy_in.dst = tl_in[k + pp_skip];
        bmk1880v2_tdma_g2l_bf16_tensor_copy(bk_ctx, &p_copy_in);

        // Change src head addr
        bm_src_addr[k] += 1 * jump_src * fmt_input_vec[k + pp_skip].getTGFmtSize();
      }
    }

    for (size_t pp = 0; pp < m_slice_info.ping_pong_size; pp++) {
      operation(ctx, bk_ctx, pp);
    }

    // tl2tg
    for (size_t pp = 0; pp < m_slice_info.ping_pong_size; pp++) {
      u32 tl_idx = tl_out.size() / m_slice_info.ping_pong_size;
      u32 pp_skip = pp * tl_idx;
      for (size_t k = 0; k < tl_idx; k++) {
        tg_out.start_address = bm_dest_addr[k];
        tg_out.shape.n = tl_out[k + pp_skip]->shape.n;
        tg_out.shape.c = tl_out[k + pp_skip]->shape.c;
        tg_out.shape.h = tl_out[k + pp_skip]->shape.h;
        tg_out.shape.w = tl_out[k + pp_skip]->shape.w;
        tg_out.fmt = fmt_output_vec[k + pp_skip].getTGFmt();
        tg_out.stride = output_stride_vec[k + pp_skip];
        bmk1880v2_tensor_lmem_t out_shape;
        bmk1880v2_tdma_l2tg_tensor_copy_param_t p_copy_out;
        p_copy_out.src = tl_out[k + pp_skip];
        p_copy_out.dst = &tg_out;
        bmk1880v2_tdma_l2g_bf16_tensor_copy(bk_ctx, &p_copy_out);

        // Change dest head addr
        bm_dest_addr[k] += 1 * jump_dst * fmt_output_vec[k + pp_skip].getTGFmtSize();
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
    u32 tl_idx = tl_in.size() / m_slice_info.ping_pong_size;
    for (size_t k = 0; k < tl_idx; k++) {
      tg_in.start_address = bm_src_addr[k];
      tg_in.shape.n = tl_in[k]->shape.n;
      tg_in.shape.c = tl_in[k]->shape.c;
      tg_in.shape.h = tl_in[k]->shape.h;
      tg_in.shape.w = tl_in[k]->shape.w;
      tg_in.fmt = fmt_input_vec[k].getTGFmt();
      tg_in.stride = input_stride_vec[k];
      bmk1880v2_tdma_tg2l_tensor_copy_param_t p_copy_in;
      p_copy_in.src = &tg_in;
      p_copy_in.dst = tl_in[k];
      bmk1880v2_tdma_g2l_bf16_tensor_copy(bk_ctx, &p_copy_in);

      // Change src head addr
      bm_src_addr[k] += 1 * jump_src * fmt_input_vec[k].getTGFmtSize();
    }

    operation(ctx, bk_ctx, 0);

    // tl2tg
    tl_idx = tl_out.size() / m_slice_info.ping_pong_size;
    for (size_t k = 0; k < tl_idx; k++) {
      tg_out.start_address = bm_dest_addr[k];
      tg_out.fmt = fmt_output_vec[k].getTGFmt();
      tg_out.shape.n = tl_out[k]->shape.n;
      tg_out.shape.c = tl_out[k]->shape.c;
      tg_out.shape.h = tl_out[k]->shape.h;
      tg_out.shape.w = tl_out[k]->shape.w;
      tg_out.stride = output_stride_vec[k];
      bmk1880v2_tensor_lmem_t out_shape;
      bmk1880v2_tdma_l2tg_tensor_copy_param_t p_copy_out;
      p_copy_out.src = tl_out[k];
      p_copy_out.dst = &tg_out;
      bmk1880v2_tdma_l2g_bf16_tensor_copy(bk_ctx, &p_copy_out);

      // Change dest head addr
      bm_dest_addr[k] += 1 * jump_dst * fmt_output_vec[k].getTGFmtSize();
    }
  }
  int ret = BM_SUCCESS;
  for (size_t k = 0; k < input.size(); k++) {
    u64 bm_start_addr = input[k].GetPAddr();
    u64 jumped_value = bm_src_addr[k] - bm_start_addr;
    u64 total_addr = input[k].m_tg.stride.n;
    if (jumped_value != total_addr) {
      printf("Error! Input %lu jumped value not align to image size %lu, original %lu\n", k,
             jumped_value, total_addr);
      ret = BM_ERR_FAILURE;
    }
  }
  for (size_t k = 0; k < output->size(); k++) {
    u64 bm_des_addr = (*output)[k].GetPAddr();
    u64 jumped_value = bm_dest_addr[k] - bm_des_addr;
    u64 total_addr = (*output)[k].m_tg.stride.n;
    if (jumped_value != total_addr) {
      printf("Error! Output %lu jumped value not align to image size %lu, original %lu\n", k,
             jumped_value, total_addr);
      ret = BM_ERR_FAILURE;
    }
  }
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