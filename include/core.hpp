#pragma once
#include "tpu_data.hpp"
#include "utils.hpp"

#include <string.h>
#include <iostream>
#include <vector>

enum IVETLType { DATA, KERNEL, TABLE };

class IveCore {
 public:
  IveCore();
  const unsigned int getNpuNum() const { return m_chip_info.npu_num; }
  virtual int init(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx) = 0;
  int run(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx, std::vector<CviImg> &input,
          std::vector<CviImg> *output, bool legacy_mode = false);

 protected:
  bmk1880v2_tensor_lmem_t *allocTLMem(bmk1880v2_context_t *bk_ctx,
                                      bmk1880v2_tensor_lmem_shape_t tl_shape, fmt_t fmt,
                                      int eu_align, IVETLType type = IVETLType::DATA);
  virtual int sliceSetup(SliceRes &slice_res, SliceRes *tg_in_res, SliceRes *tg_out_res);
  virtual int runSetup(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx,
                       const std::vector<bmk1880v2_tensor_tgmem_shape_t> &tg_in_slices,
                       const std::vector<bmk1880v2_tensor_tgmem_shape_t> &tg_out_slices,
                       std::vector<u32> *tl_in_idx, std::vector<u32> *tl_out_idx,
                       const bool enable_cext) = 0;
  virtual void operation(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx, u32 ping_idx) = 0;
  virtual int postProcess(bmctx_t *ctx);

  u32 m_nums_of_input = 1;
  u32 m_nums_of_output = 1;
  SliceInfo m_slice_info;
  kernelInfo m_kernel_info;
  std::vector<IVETLType> m_tl_type;
  std::vector<bmk1880v2_tensor_lmem_t *> m_tl_vec;
  std::string m_cmdbuf_subfix;

 private:
  int getSlice(const u32 nums_of_lmem, const u32 nums_of_table, const u32 fixed_lmem_size,
               const u32 n, const u32 c, const u32 h, const u32 w, const u32 table_size,
               const kernelInfo kernel_info, const int npu_num, sliceUnit *unit_h,
               sliceUnit *unit_w, const bool enable_cext);
  int freeTLMems(bmk1880v2_context_t *bk_ctx);
  int runSingleSizeKernel(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx, std::vector<CviImg> &input,
                          std::vector<CviImg> *output, bool enable_min_max = false);
  int runSingleSizeExtKernel(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx, std::vector<CviImg> &input,
                             std::vector<CviImg> *output, bool enable_min_max = false);
  int runNoKernel(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx, std::vector<CviImg> &input,
                  std::vector<CviImg> *output, bool enable_min_max = false);

  bool m_write_cmdbuf = false;
  cvi_chip_info_s m_chip_info;
  u32 m_table_per_channel_size = 0;
};