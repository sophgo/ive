#ifndef _IVE_EXPERIMENTAL_H
#define _IVE_EXPERIMENTAL_H
#include "ive.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Convert IVE_IMAGE_S to VIDEO_FRAME_S. Note that do not unmap the virtual address from
 *        IVE_IMAGE_S.
 *
 * @param pstIISrc IVE_IMAGE_S input.
 * @param pstVFDst VIDEO_FRAME_S output.
 * @return CVI_S32 Return CVI_SUCCESS if operation succeeded.
 */
CVI_S32 CVI_IVE_Image2VideoFrame(IVE_IMAGE_S *pstIISrc, VIDEO_FRAME_S *pstVFDst);

/**
 * @brief Multiply every pixel with given image into one value.
 *
 * @param pIveHandle Ive instance handler.
 * @param pstImg Input image, only accepts U8C1 or BF16C1.
 * @param sum Multiply result.
 * @param bInstant Dummy variable.
 * @return CVI_S32 Return CVI_SUCCESS if succeed.
 */
CVI_S32 CVI_IVE_MulSum(IVE_HANDLE pIveHandle, IVE_IMAGE_S *pstImg, double *sum, bool bInstant);

#ifdef __cplusplus
}
#endif

#endif  // End of _IVE_EXPERIMENTAL_H