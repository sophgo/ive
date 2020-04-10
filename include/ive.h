#ifndef _IVE_H
#define _IVE_H
#include "cvi_comm_ive.h"
#ifndef __cplusplus
#include <stdbool.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif
/**
 * @brief Direct GLOG notes to file instead of stdio.
 *
 * @param argv0 char array from main function.
 */
void CVI_SYS_LOGGING(char *argv0);

/**
 * @brief Create an IVE instance handler.
 *
 * @return IVE_HANDLE An IVE instance handle.
 */
IVE_HANDLE CVI_IVE_CreateHandle();

/**
 * @brief Destroy an Ive instance handler.
 *
 * @param pIveHandle Ive instanace handler.
 * @return CVI_S32 Return CVI_SUCCESS if instance is successfully destroyed.
 */
CVI_S32 CVI_IVE_DestroyHandle(IVE_HANDLE pIveHandle);

/**
 * @brief Flush cache data to RAM. Call this after IVE_IMAGE_S VAddr operations.
 *
 * @param pIveHandle Ive instanace handler.
 * @param pstImg Image to be flushed.
 * @return CVI_S32 Return CVI_SUCCESS if operation succeeded.
 */
CVI_S32 CVI_IVE_BufFlush(IVE_HANDLE pIveHandle, IVE_IMAGE_S *pstImg);

/**
 * @brief Update cache from RAM. Call this function before using data from VAddr \
 *        in CPU.
 *
 * @param pIveHandle Ive instanace handler.
 * @param pstImg Cache image to be updated.
 * @return CVI_S32 Return CVI_SUCCESS if operation succeeded.
 */
CVI_S32 CVI_IVE_BufRequest(IVE_HANDLE pIveHandle, IVE_IMAGE_S *pstImg);

/**
 * @brief Flush the TPU command buffer saved inside the instnace handler.
 *
 * @param pIveHandle Ive instance handler.
 * @return CVI_S32 Return CVI_SUCCESS if operation succeed.
 */
CVI_S32 CVI_IVE_CmdFlush(IVE_HANDLE pIveHandle);

/**
 * @brief Create a IVE_MEM_INFO_S.
 *
 * @param pIveHandle Ive instance handler.
 * @param pstMemInfo The input mem info structure.
 * @param u32Size The size of the mem info in 1d.
 * @return CVI_S32
 */
CVI_S32 CVI_IVE_CreateMemInfo(IVE_HANDLE pIveHandle, IVE_MEM_INFO_S *pstMemInfo, CVI_U32 u32Size);

/**
 * @brief Create an IVE_IMAGE_S.
 *
 * @param pIveHandle Ive instance handler.
 * @param pstImg The input image stucture.
 * @param enType The image type. e.g. IVE_IMAGE_TYPE_U8C1.
 * @param u16Width The image width.
 * @param u16Height The image height.
 * @return CVI_S32 Return CVI_SUCCESS if operation succeed.
 */
CVI_S32 CVI_IVE_CreateImage(IVE_HANDLE pIveHandle, IVE_IMAGE_S *pstImg, IVE_IMAGE_TYPE_E enType,
                            CVI_U16 u16Width, CVI_U16 u16Height);

/**
 * @brief Get the sub image from an image with the given coordiantes. The data is shared without
 * copy.
 *
 * @param pIveHandle Ive instance handler.
 * @param pstSrc The input image.
 * @param pstDst The output sub image.
 * @param u16X1 The X1 coordinate from original image.
 * @param u16Y1 The Y1 coordinate from original image.
 * @param u16X2 The X2 coordinate from original image.
 * @param u16Y2 The Y2 coordinate from original image.
 * @return CVI_S32 Return CVI_SUCCESS if operation succeed.
 */
CVI_S32 CVI_IVE_SubImage(IVE_HANDLE pIveHandle, IVE_SRC_IMAGE_S *pstSrc, IVE_DST_IMAGE_S *pstDst,
                         CVI_U16 u16X1, CVI_U16 u16Y1, CVI_U16 u16X2, CVI_U16 u16Y2);

/**
 * @brief Read an image from file system.
 *
 * @param pIveHandle Ive instance handler.
 * @param filename File path to the image.
 * @param enType Type of the destination image.
 * @return IVE_IMAGE_S Return IVE_IMAGE_S
 */
IVE_IMAGE_S CVI_IVE_ReadImage(IVE_HANDLE pIveHandle, const char *filename, IVE_IMAGE_TYPE_E enType);

/**
 * @brief Write an IVE_IMAGE_S to file system.
 *
 * @param pIveHandle Ive instance handler.
 * @param filename Save file path.
 * @param pstImg Input image.
 * @return CVI_S32 Return CVI_SUCCESS if succeed.
 */
CVI_S32 CVI_IVE_WriteImage(IVE_HANDLE pIveHandle, const char *filename, IVE_IMAGE_S *pstImg);

/**
 * @brief Free Allocated IVE_MEM_INFO_S.
 *
 * @param pIveHandle Ive instance handler.
 * @param pstMemInfo Allocated IVE_MEM_INFO_S.
 * @return CVI_S32 Return CVI_SUCCESS.
 */
CVI_S32 CVI_SYS_FreeM(IVE_HANDLE pIveHandle, IVE_MEM_INFO_S *pstMemInfo);

/**
 * @brief Free Allocated IVE_IMAGE_S.
 *
 * @param pIveHandle Ive instance handler.
 * @param pstMemInfo Allocated IVE_IMAGE_S.
 * @return CVI_S32 Return CVI_SUCCESS.
 */
CVI_S32 CVI_SYS_FreeI(IVE_HANDLE pIveHandle, IVE_IMAGE_S *pstImg);

/**
 * @brief Copy data between image.
 *
 * @param pIveHandle Ive instance handler.
 * @param pstSrc Input image.
 * @param pstDst Output image.
 * @param pstDmaCtrl Dma control parameters.
 * @param bInstant Dummy variable.
 * @return CVI_S32 Return CVI_SUCCESS if succeed.
 */
CVI_S32 CVI_IVE_DMA(IVE_HANDLE pIveHandle, IVE_SRC_IMAGE_S *pstSrc, IVE_DST_IMAGE_S *pstDst,
                    IVE_DMA_CTRL_S *pstDmaCtrl, bool bInstant);

/**
 * @brief Convert image to different image type.
 *
 * @param pIveHandle Ive instance handler.
 * @param pstSrc Input image.
 * @param pstDst Outpu image.
 * @param pstItcCtrl Convert parameters.
 * @param bInstant Dummy variable.
 * @return CVI_S32 Return CVI_SUCCESS if succeed.
 */
CVI_S32 CVI_IVE_ImageTypeConvert(IVE_HANDLE pIveHandle, IVE_SRC_IMAGE_S *pstSrc,
                                 IVE_DST_IMAGE_S *pstDst, IVE_ITC_CRTL_S *pstItcCtrl,
                                 bool bInstant);

/**
 * @brief Add two image and output the result.
 *
 * @param pIveHandle Ive instance handler.
 * @param pstSrc1 Input image 1.
 * @param pstSrc2 Input image 2.
 * @param pstDst Output result.
 * @param ctrl Add control parameter.
 * @param bInstant Dummy variable.
 * @return CVI_S32 Return CVI_SUCCESS if succeed.
 */
CVI_S32 CVI_IVE_Add(IVE_HANDLE pIveHandle, IVE_SRC_IMAGE_S *pstSrc1, IVE_SRC_IMAGE_S *pstSrc2,
                    IVE_DST_IMAGE_S *pstDst, IVE_ADD_CTRL_S *ctrl, bool bInstant);

/**
 * @brief AND two images and output the result.
 *
 * @param pIveHandle Ive instance handler.
 * @param pstSrc1 Input image 1.
 * @param pstSrc2 Input image 2.
 * @param pstDst Output result.
 * @param bInstant Dummy variable.
 * @return CVI_S32 CVI_S32 Return CVI_SUCCESS if succeed.
 */
CVI_S32 CVI_IVE_And(IVE_HANDLE pIveHandle, IVE_SRC_IMAGE_S *pstSrc1, IVE_SRC_IMAGE_S *pstSrc2,
                    IVE_DST_IMAGE_S *pstDst, bool bInstant);

/**
 * @brief Calculate the average of the sliced cells of an image. The output image size will be \
 *        (input_w / u32CellSize, output_h / u32CellSize)
 *
 * @param pIveHandle Ive instance handler.
 * @param pstSrc Input image. Only accepts U8C1.
 * @param pstDst Output result.
 * @param pstBlkCtrl Block control parameter.
 * @param bInstant Dummy variable.
 * @return CVI_S32 CVI_S32 Return CVI_SUCCESS if succeed.
 */
CVI_S32 CVI_IVE_BLOCK(IVE_HANDLE pIveHandle, IVE_SRC_IMAGE_S *pstSrc, IVE_DST_IMAGE_S *pstDst,
                      IVE_BLOCK_CTRL_S *pstBlkCtrl, bool bInstant);

/**
 * @brief Dilate a gray scale image.
 *
 * @param pIveHandle Ive instance handler.
 * @param pstSrc Input image. Only accepts U8C1.
 * @param pstDst Outpu result.
 * @param pstDilateCtrl Dilate control parameter.
 * @param bInstant Dummy variable.
 * @return CVI_S32 CVI_S32 Return CVI_SUCCESS if succeed.
 */
CVI_S32 CVI_IVE_Dilate(IVE_HANDLE pIveHandle, IVE_SRC_IMAGE_S *pstSrc, IVE_DST_IMAGE_S *pstDst,
                       IVE_DILATE_CTRL_S *pstDilateCtrl, bool bInstant);

/**
 * @brief Erode a gray scale image.
 *
 * @param pIveHandle Ive instance handler.
 * @param pstSrc Input image. Only accepts U8C1.
 * @param pstDst Output result.
 * @param pstErodeCtrl Erode control variable.
 * @param bInstant Dummy variable.
 * @return CVI_S32 CVI_S32 Return CVI_SUCCESS if succeed.
 */
CVI_S32 CVI_IVE_Erode(IVE_HANDLE pIveHandle, IVE_SRC_IMAGE_S *pstSrc, IVE_DST_IMAGE_S *pstDst,
                      IVE_ERODE_CTRL_S *pstErodeCtrl, bool bInstant);

/**
 * @brief Apply a filter to an image.
 *
 * @param pIveHandle Ive instance handler.
 * @param pstSrc Input image.
 * @param pstDst Output result.
 * @param pstFltCtrl Filter control parameter.
 * @param bInstant Dummy variable.
 * @return CVI_S32 Return CVI_SUCCESS if succeed.
 */
CVI_S32 CVI_IVE_Filter(IVE_HANDLE pIveHandle, IVE_SRC_IMAGE_S *pstSrc, IVE_DST_IMAGE_S *pstDst,
                       IVE_FILTER_CTRL_S *pstFltCtrl, bool bInstant);

/**
 * @brief Get size of the HOG histogram.
 *
 * @param u16Width Input image width.
 * @param u16Height Input image height.
 * @param u8BinSize Bin size.
 * @param u16CellSize Cell size.
 * @param u16BlkSize  Block size.
 * @param u16BlkStepX Block step in X dimension.
 * @param u16BlkStepY Block step in Y dimension.
 * @param u32HogSize Output HOG size (length * sizeof(u32)).
 * @return CVI_S32 Return CVI_SUCCESS if succeed.
 */
CVI_S32 CVI_IVE_GET_HOG_SIZE(CVI_U16 u16Width, CVI_U16 u16Height, CVI_U8 u8BinSize,
                             CVI_U16 u16CellSize, CVI_U16 u16BlkSize, CVI_U16 u16BlkStepX,
                             CVI_U16 u16BlkStepY, CVI_U32 *u32HogSize);

/**
 * @brief Calculate the HOG of an image. The gradient calculation uses Sobel gradient.
 *
 * @param pIveHandle Ive instance handler.
 * @param pstSrc Input image.
 * @param pstDstH Output horizontal gradient result.
 * @param pstDstV Output vertical gradient result.
 * @param pstDstMag Output L2 Norm magnitude result from Gradient V, H.
 * @param pstDstAng Output atan2 angular result from Gradient V / Gradient H.
 * @param pstDstHist HOG histogram. result.
 * @param pstHogCtrl HOG control parameter.
 * @param bInstant DUmmy variable.
 * @return CVI_S32 CVI_S32 Return CVI_SUCCESS if succeed.
 */
CVI_S32 CVI_IVE_HOG(IVE_HANDLE pIveHandle, IVE_SRC_IMAGE_S *pstSrc, IVE_DST_IMAGE_S *pstDstH,
                    IVE_DST_IMAGE_S *pstDstV, IVE_DST_IMAGE_S *pstDstMag,
                    IVE_DST_IMAGE_S *pstDstAng, IVE_DST_MEM_INFO_S *pstDstHist,
                    IVE_HOG_CTRL_S *pstHogCtrl, bool bInstant);

/**
 * @brief Calculate the Magnitude and Nagular result from given horizontal and vertical gradients.
 *
 * @param pIveHandle Ive instance handler.
 * @param pstSrcH Input horizontal gradient.
 * @param pstSrcV Input vertical gradient.
 * @param pstDstMag Output L2 norm magnitude result.
 * @param pstDstAng Output atan2 angular result.
 * @param pstMaaCtrl Magnitude and angular control parameter.
 * @param bInstant Dummy variable.
 * @return CVI_S32 CVI_S32 Return CVI_SUCCESS if succeed.
 */
CVI_S32 CVI_IVE_MagAndAng(IVE_HANDLE pIveHandle, IVE_SRC_IMAGE_S *pstSrcH, IVE_SRC_IMAGE_S *pstSrcV,
                          IVE_DST_IMAGE_S *pstDstMag, IVE_DST_IMAGE_S *pstDstAng,
                          IVE_MAG_AND_ANG_CTRL_S *pstMaaCtrl, bool bInstant);

/**
 * @brief Map src image to dst image with a given table.
 *
 * @param pIveHandle Ive instance handler.
 * @param pstSrc Input image.
 * @param pstMap Mapping table. (length 256.)
 * @param pstDst Output image.
 * @param bInstant Dummy variable.
 * @return CVI_S32 Return CVI_SUCCESS if succeed.
 */
CVI_S32 CVI_IVE_Map(IVE_HANDLE pIveHandle, IVE_SRC_IMAGE_S *pstSrc, IVE_MEM_INFO_S *pstMap,
                    IVE_DST_IMAGE_S *pstDst, bool bInstant);

/**
 * @brief Calculate the normalized gradient of an image.
 *
 * @param pIveHandle Ive instance handler.
 * @param pstSrc Input image. Only accepts U8C1.
 * @param pstDstH Output horizontal gradient result. Accepts U8C1, S8C1.
 * @param pstDstV Output vertical gradient result. Accepts U8C1, S8C1.
 * @param pstDstHV Output combined L2 norm gradient result.
 * @param pstNormGradCtrl Norm gradient control parameter.
 * @param bInstant Dummy variable.
 * @return CVI_S32 CVI_S32 Return CVI_SUCCESS if succeed.
 */
CVI_S32 CVI_IVE_NormGrad(IVE_HANDLE pIveHandle, IVE_SRC_IMAGE_S *pstSrc, IVE_DST_IMAGE_S *pstDstH,
                         IVE_DST_IMAGE_S *pstDstV, IVE_DST_IMAGE_S *pstDstHV,
                         IVE_NORM_GRAD_CTRL_S *pstNormGradCtrl, bool bInstant);

/**
 * @brief Or two images and output the result.
 *
 * @param pIveHandle Ive instance handler.
 * @param pstSrc1 Input image 1.
 * @param pstSrc2 Input image 2.
 * @param pstDst Output result.
 * @param bInstant Dummy variable.
 * @return CVI_S32 CVI_S32 Return CVI_SUCCESS if succeed.
 */
CVI_S32 CVI_IVE_Or(IVE_HANDLE pIveHandle, IVE_SRC_IMAGE_S *pstSrc1, IVE_SRC_IMAGE_S *pstSrc2,
                   IVE_DST_IMAGE_S *pstDst, bool bInstant);

/**
 * @brief Run sigmoid of an input image.
 *
 * @param pIveHandle Ive instance handler.
 * @param pstSrc Input image.
 * @param pstDst Output image.
 * @param bInstant Dummy variable.
 * @return CVI_S32 CVI_S32 Return CVI_SUCCESS if succeed.
 */
CVI_S32 CVI_IVE_Sigmoid(IVE_HANDLE pIveHandle, IVE_SRC_IMAGE_S *pstSrc, IVE_DST_IMAGE_S *pstDst,
                        bool bInstant);

/**
 * @brief Calculate SAD result with two same size input image.
 *
 * @param pIveHandle Ive instance handler.
 * @param pstSrc1 Input image 1.
 * @param pstSrc2 Input image 2.
 * @param pstSad Output SAD result.
 * @param pstThr Output thresholded SAD result.
 * @param pstSadCtrl SAD control parameter.
 * @param bInstant Dummy variable.
 * @return CVI_S32 CVI_S32 Return CVI_SUCCESS if succeed.
 */
CVI_S32 CVI_IVE_SAD(IVE_HANDLE pIveHandle, IVE_SRC_IMAGE_S *pstSrc1, IVE_SRC_IMAGE_S *pstSrc2,
                    IVE_DST_IMAGE_S *pstSad, IVE_DST_IMAGE_S *pstThr, IVE_SAD_CTRL_S *pstSadCtrl,
                    bool bInstant);

/**
 * @brief Calculate horizontal and vertical gradient using Sobel kernel.
 *
 * @param pIveHandle Ive instance handler.
 * @param pstSrc Input image. Only accepts U8C1.
 * @param pstDstH Output horizontal gradient result.
 * @param pstDstV Output vertical gradient result.
 * @param pstSobelCtrl Sobel control parameter.
 * @param bInstant Dummy variable.
 * @return CVI_S32 CVI_S32 Return CVI_SUCCESS if succeed.
 */
CVI_S32 CVI_IVE_Sobel(IVE_HANDLE pIveHandle, IVE_SRC_IMAGE_S *pstSrc, IVE_DST_IMAGE_S *pstDstH,
                      IVE_DST_IMAGE_S *pstDstV, IVE_SOBEL_CTRL_S *pstSobelCtrl, bool bInstant);

/**
 * @brief Subtract two images and output the result.
 *
 * @param pIveHandle Ive instance handler.
 * @param pstSrc1 Input image 1.
 * @param pstSrc2 Input image 2.
 * @param pstDst Output result.
 * @param ctrl Subtract control parameter.
 * @param bInstant Dummy variable.
 * @return CVI_S32 CVI_S32 Return CVI_SUCCESS if succeed.
 */
CVI_S32 CVI_IVE_Sub(IVE_HANDLE pIveHandle, IVE_SRC_IMAGE_S *pstSrc1, IVE_SRC_IMAGE_S *pstSrc2,
                    IVE_DST_IMAGE_S *pstDst, IVE_SUB_CTRL_S *ctrl, bool bInstant);

/**
 * @brief Calculate the threshold result of an image.
 *
 * @param pIveHandle Ive instance handler.
 * @param pstSrc Input image. Only accepts U8C1.
 * @param pstDst Output result.
 * @param ctrl Threshold control parameter.
 * @param bInstant Dummy variable.
 * @return CVI_S32 CVI_S32 Return CVI_SUCCESS if succeed.
 */
CVI_S32 CVI_IVE_Thresh(IVE_HANDLE pIveHandle, IVE_SRC_IMAGE_S *pstSrc, IVE_DST_IMAGE_S *pstDst,
                       IVE_THRESH_CTRL_S *ctrl, bool bInstant);

/**
 * @brief Threshold an S16 image with high low threshold to U8 or S8.
 *
 * @param pIveHandle Ive instance handler.
 * @param pstSrc Input S16 image.
 * @param pstDst Output U8/ S8 image.
 * @param pstThrS16Ctrl S16 threshold control parameter.
 * @param bInstant Dummy variable.
 * @return CVI_S32 Return CVI_SUCCESS if succeed.
 */
CVI_S32 CVI_IVE_Thresh_S16(IVE_HANDLE pIveHandle, IVE_SRC_IMAGE_S *pstSrc, IVE_DST_IMAGE_S *pstDst,
                           IVE_THRESH_S16_CTRL_S *pstThrS16Ctrl, bool bInstant);

/**
 * @brief Threshold an U16 image with high low threshold to U8.
 *
 * @param pIveHandle Ive instance handler.
 * @param pstSrc Input U16 image.
 * @param pstDst Output U8 image.
 * @param pstThrU16Ctrl U16 threshold control parameter.
 * @param bInstant Dummy variable.
 * @return CVI_S32 Return CVI_SUCCESS if succeed.
 */
CVI_S32 CVI_IVE_Thresh_U16(IVE_HANDLE pIveHandle, IVE_SRC_IMAGE_S *pstSrc, IVE_DST_IMAGE_S *pstDst,
                           IVE_THRESH_U16_CTRL_S *pstThrU16Ctrl, bool bInstant);

/**
 * @brief XOR two images and output the result.
 *
 * @param pIveHandle Ive instance handler.
 * @param pstSrc1 Input image 1.
 * @param pstSrc2 Input image 2.
 * @param pstDst Output result.
 * @param bInstant Dummy variable.
 * @return CVI_S32 CVI_S32 Return CVI_SUCCESS if succeed.
 */
CVI_S32 CVI_IVE_Xor(IVE_HANDLE pIveHandle, IVE_SRC_IMAGE_S *pstSrc1, IVE_SRC_IMAGE_S *pstSrc2,
                    IVE_DST_IMAGE_S *pstDst, bool bInstant);


// for cpu version

/**
 * @brief INTEG make a integral image with one gray image 
 *
 * @param pIveHandle Ive instance handler.
 * @param pstSrc Input gray image.
 * @param pstDst Output int image.
 * @param bInstant Dummy variable.
 * @return CVI_S32 CVI_S32 Return CVI_SUCCESS if succeed.
 */

CVI_S32 CVI_IVE_Integ(IVE_HANDLE pIveHandle, IVE_SRC_IMAGE_S *pstSrc, IVE_DST_MEM_INFO_S *pstDst, 
                      IVE_INTEG_CTRL_S *ctrl, bool bInstant);

CVI_S32 CVI_IVE_Hist(IVE_HANDLE pIveHandle, IVE_SRC_IMAGE_S *pstSrc,
                     IVE_DST_MEM_INFO_S *pstDst, bool bInstant); 


CVI_S32 CVI_IVE_EqualizeHist(IVE_HANDLE pIveHandle, IVE_SRC_IMAGE_S *pstSrc, 
                             IVE_DST_IMAGE_S *pstDst, IVE_EQUALIZE_HIST_CTRL_S *ctrl, bool bInstant);


CVI_S32 CVI_IVE_16BitTo8Bit(IVE_HANDLE pIveHandle, IVE_SRC_IMAGE_S *pstSrc, 
                            IVE_DST_IMAGE_S *pstDst, IVE_16BIT_TO_8BIT_CTRL_S *ctrl, bool bInstant);


CVI_S32 CVI_IVE_NCC(IVE_HANDLE pIveHandle, IVE_SRC_IMAGE_S *pstSrc1,
                    IVE_SRC_IMAGE_S *pstSrc2, IVE_DST_MEM_INFO_S *pstDst, bool bInstant);

#ifdef __cplusplus
}
#endif

#endif  // End of _IVE_H