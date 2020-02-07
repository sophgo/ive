#include "tpu/tpu_block.hpp"

#include <glog/logging.h>
#include "bmkernel/bm1880v2/1880v2_fp_convert.h"

// clang-format off
static char test_array[] = {
  1, 2, 3, 4, 5, 2, 2, 2, 2, 2, 7, 7, 7, 7, 7, 9, 9, 9, 9, 9, 1, 2, 3, 4, 5,
  1, 2, 3, 4, 5, 2, 2, 2, 2, 2, 7, 7, 7, 7, 7, 9, 9, 9, 9, 9, 1, 2, 3, 4, 5,
  1, 2, 3, 4, 5, 2, 2, 2, 2, 2, 7, 7, 7, 7, 7, 9, 9, 9, 9, 9, 1, 2, 3, 4, 5,
  1, 2, 3, 4, 5, 2, 2, 2, 2, 2, 7, 7, 7, 7, 7, 9, 9, 9, 9, 9, 1, 2, 3, 4, 5,
  1, 2, 3, 4, 5, 2, 2, 2, 2, 2, 7, 7, 7, 7, 7, 9, 9, 9, 9, 9, 1, 2, 3, 4, 5, //
  6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 7, 7, 7, 7, 7,
  6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 7, 7, 7, 7, 7,
  6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 7, 7, 7, 7, 7,
  6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 7, 7, 7, 7, 7,
  6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 7, 7, 7, 7, 7, //
  1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 1, 1, 1, 1, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
  1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 1, 1, 1, 1, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
  1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 1, 1, 1, 1, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
  1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 1, 1, 1, 1, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
  1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 1, 1, 1, 1, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, //
  1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
  1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
  1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
  1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
  1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, //
  1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 8, 8, 8, 8, 8, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
  1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 8, 8, 8, 8, 8, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
  1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 8, 8, 8, 8, 8, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
  1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 8, 8, 8, 8, 8, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
  1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 8, 8, 8, 8, 8, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, //
};
//clang-format on

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  // Create instance
  bmctx_t ctx;
  bmk1880v2_context_t* bk_ctx;
  createHandle(&ctx, &bk_ctx);
  printf("BM Kernel init.\n");

  CviImg cvi_img(&ctx, 1, 25, 25, FMT_U8);
  memcpy(cvi_img.GetVAddr(), test_array, 625);
  CviImg result(&ctx, 1, 5, 5, FMT_U8);
  CviImg result2(&ctx, 1, 5, 5, FMT_BF16);

  std::vector<CviImg> inputs, outputs;
  inputs.emplace_back(cvi_img);
  outputs.emplace_back(result);

  printf("Run TPU Block.\n");
  // Run TPU Add
  IveTPUBlock tpu_block;
  tpu_block.setCellSize(5, 1);
  tpu_block.init(&ctx, bk_ctx);
  tpu_block.runSingleSizeKernel(&ctx, bk_ctx, inputs, &outputs);

  printf("Result:\n");
  for (size_t i = 0; i < 5; i++) {
    for (size_t j = 0; j < 5; j++) {
      printf("%d ", result.GetVAddr()[i * 5 + j]);
    }
    printf("\n");
  }

  outputs.clear();
  outputs.emplace_back(result2);
  IveTPUBlockBF16 tpu_block_bf16;
  tpu_block_bf16.setCellSize(5, 1);
  tpu_block_bf16.init(&ctx, bk_ctx);
  tpu_block_bf16.runSingleSizeKernel(&ctx, bk_ctx, inputs, &outputs);

  printf("Result:\n");
  for (size_t i = 0; i < 5; i++) {
    for (size_t j = 0; j < 5; j++) {
      printf("%f ", convert_bf16_fp32(((short*)result2.GetVAddr())[i * 5 + j]));
    }
    printf("\n");
  }
  // Free memory, instance
  cvi_img.Free(&ctx);
  result.Free(&ctx);
  result2.Free(&ctx);
  destroyHandle(&ctx);
  return 0;
}
