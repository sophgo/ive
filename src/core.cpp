#include "core.hpp"

#include <iostream>
#include "debug.hpp"

#define BM1880V2_LMEM_SIZE 32768

int IveCore::getSlice(const u32 nums_of_lmem, const u32 table_size_per_channel,
                      const u32 fixed_lmem_size, const u32 n, const u32 c, const u32 h, const u32 w,
                      const kernelInfo kernel_info, sliceUnit *unit_h, sliceUnit *unit_w) {
  // Calculate fixed kernel size
  u32 kernel_sz = (m_kernel_info.nums_of_kernel * m_kernel_info.size * m_kernel_info.size +
                   MULTIPLIER_ONLY_PACKED_DATA_SIZE * m_kernel_info.use_multiplier);
  // Find max available mem for one tl.
  const u32 available_lmem_per_tl =
      (BM1880V2_LMEM_SIZE - ((kernel_sz + table_size_per_channel) * c) - fixed_lmem_size) /
      nums_of_lmem;
  u32 w_length = w;
  u32 h_tmp_slice = 0;
  int w_num = 1;
  // Here the default value for kernel size is 1. The h_slice should never smaller than kernel size.
  while (h_tmp_slice < m_kernel_info.size) {
    w_length = w / w_num;
    h_tmp_slice = available_lmem_per_tl / c / w_length;
    w_num++;
  }
  unit_h->slice = h_tmp_slice;
  unit_h->skip = unit_h->slice - kernel_info.size + 1;
  unit_h->left = (h - kernel_info.pad[2] - kernel_info.pad[3]) % unit_h->skip;
  unit_h->turn = ceil((float)(h - kernel_info.pad[2] - kernel_info.pad[3]) / unit_h->skip);
  unit_w->slice = w_length;
  unit_w->skip = unit_w->slice - kernel_info.size + 1;
  unit_w->left = (w - kernel_info.pad[0] - kernel_info.pad[1]) % unit_w->skip;
  unit_w->turn = ceil((float)(w - kernel_info.pad[0] - kernel_info.pad[1]) / unit_w->skip);
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

int IveCore::runSingleSizeKernel(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx,
                                 std::vector<CviImg> &input, std::vector<CviImg> *output) {
  u32 batch = input[0].m_tg.shape.n;
  u32 channel = input[0].m_tg.shape.c;
  u32 height = input[0].m_tg.shape.h;
  u32 width = input[0].m_tg.shape.w;
  // FIXME: Currently only supports same input / ouput size
  // Get slice and create block shape size
  getSlice(m_slice_info.nums_of_tl, m_slice_info.table_size_per_channel, m_slice_info.fix_lmem_size,
           batch, channel, height, width, m_kernel_info, &m_slice_info.h, &m_slice_info.w);
  // TODO: Add a virtual function here if future need multi-size tls?
  std::vector<bmk1880v2_tensor_tgmem_shape_t> s_in_vec, s_out_vec;
  for (size_t k = 0; k < input.size(); k++) {
    s_in_vec.push_back({1, channel, m_slice_info.h.slice, m_slice_info.w.slice});
  }
  for (size_t k = 0; k < output->size(); k++) {
    s_out_vec.push_back({1, channel, m_slice_info.h.slice, m_slice_info.w.slice});
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
  // FIXME: Currently input/ output use same shape, skip, need a mechanism to auto calculate the
  // correct value for each image.
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
    bm_dest_addr.push_back(
        bm_des_addr + ((*output)[k].m_tg.shape.w * m_kernel_info.pad[2] + m_kernel_info.pad[0]) *
                          fmt_output_vec[k].getTGFmtSize());
  }

  // Main for loop
  for (u32 i = 0; i < m_slice_info.h.turn; i++) {
    // Re-assign head address to w.
    std::vector<u64> bm_src_addr_w = bm_src_addr;
    std::vector<u64> bm_dest_addr_w = bm_dest_addr;

    for (u32 j = 0; j < m_slice_info.w.turn; j++) {
      // tg2tl
      for (size_t k = 0; k < tl_in.size(); k++) {
        tg_in.start_address = bm_src_addr_w[k];
        tg_in.shape = s_in_vec[k];
        tg_in.fmt = fmt_input_vec[k].getTGFmt();
        tg_in.stride.h = input[k].m_tg.shape.w * fmt_input_vec[k].getTGFmtSize();
        tg_in.stride.c = input[k].m_tg.shape.h * tg_in.stride.h;
        tg_in.stride.n = input[k].m_tg.shape.c * tg_in.stride.c;
        bmk1880v2_tdma_tg2l_tensor_copy_param_t p_copy_in;
        p_copy_in.src = &tg_in;
        p_copy_in.dst = tl_in[k];
        bmk1880v2_tdma_g2l_bf16_tensor_copy(bk_ctx, &p_copy_in);

        // Change src head addr
        bm_src_addr_w[k] += 1 * m_slice_info.w.skip * fmt_input_vec[k].getTGFmtSize();
      }
      bmruntime_bmkernel_submit(*ctx);

      operation(ctx, bk_ctx);

      // tl2tg
      for (size_t k = 0; k < tl_out.size(); k++) {
        tg_out.start_address = bm_dest_addr_w[k];
        tg_out.fmt = fmt_output_vec[k].getTGFmt();
        tg_out.shape.n = s_out_vec[k].n;
        tg_out.shape.c = s_out_vec[k].c;
        tg_out.shape.h = (i == m_slice_info.h.turn - 1 && m_slice_info.h.left != 0)
                             ? m_slice_info.h.left
                             : s_out_vec[k].h - (m_kernel_info.pad[2] + m_kernel_info.pad[3]);
        tg_out.shape.w = (j == m_slice_info.w.turn - 1 && m_slice_info.w.left != 0)
                             ? m_slice_info.w.left
                             : s_out_vec[k].w - (m_kernel_info.pad[0] + m_kernel_info.pad[1]);
        tg_out.stride.h = (*output)[k].m_tg.shape.w * fmt_output_vec[k].getTGFmtSize();
        tg_out.stride.c = (*output)[k].m_tg.shape.h * tg_out.stride.h;
        tg_out.stride.n = (*output)[k].m_tg.shape.c * tg_out.stride.c;
        bmk1880v2_tensor_lmem_t out_shape;
        // printf("st addr%d, tg st addr %lu\n", tl_out[k]->start_address, bm_dest_addr_w[k]);
        out_shape.start_address =
            tl_out[k]->start_address +
            (1 * tl_out[k]->shape.w * m_kernel_info.pad[2] + m_kernel_info.pad[0]) *
                fmt_output_vec[k].getTLFmtSize();
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
        bm_dest_addr_w[k] += 1 * m_slice_info.w.skip * fmt_output_vec[k].getTGFmtSize();
      }
      bmruntime_bmkernel_submit(*ctx);
    }
    // Change src/ dest head addr
    for (size_t k = 0; k < tl_in.size(); k++) {
      bm_src_addr[k] +=
          1 * input[k].m_tg.shape.w * m_slice_info.h.skip * fmt_input_vec[k].getTGFmtSize();
    }
    for (size_t k = 0; k < tl_out.size(); k++) {
      bm_dest_addr[k] +=
          1 * (*output)[k].m_tg.shape.w * m_slice_info.h.skip * fmt_output_vec[k].getTGFmtSize();
    }
  }

  IVE_DEBUG("Slice info:\n");
  IVE_DEBUG("{ h_slice, h_turn, h_skip, h_left} = { %d, %d, %d, %d}\n", m_slice_info.h.slice,
            m_slice_info.h.turn, m_slice_info.h.skip, m_slice_info.h.left);
  IVE_DEBUG("{ w_slice, w_turn, w_skip, w_left} = { %d, %d, %d, %d}\n", m_slice_info.w.slice,
            m_slice_info.w.turn, m_slice_info.w.skip, m_slice_info.w.left);

  freeTLMems(bk_ctx);
  return BM_SUCCESS;
}