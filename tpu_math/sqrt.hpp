#pragma once
#include <bmkernel/bm1880v2/bmkernel_1880v2.h>
#include <libbmruntime/bmruntime_bmkernel.h>

int bf16_emit_sqrt(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx,
	bmk1880v2_tensor_lmem_t* tl_ifmap,
	bmk1880v2_tensor_lmem_t* tl_buf,
	bmk1880v2_tensor_lmem_t* tl_ofmap_bf16,
	bmk1880v2_tensor_lmem_t* tl_ofmap_u8
);
