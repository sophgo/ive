#include <assert.h>
#include <math.h>
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include <limits.h>

#include "ive_tpu.h"

//raw12_to_uint16
void fill_raw12(unsigned char* input, int len) {
    for(int i = 0; i < len; i++) {
        input[i] = rand() % 256;
    }
}

void raw12_to_uint16_cpu(unsigned char *input_data, uint16_t *output_cpu, int w, int h) {
    const size_t pixel_cnt = (size_t)w * (size_t)h;
    const size_t group_cnt = pixel_cnt >> 1;

    size_t s_idx = 0;
    size_t d_idx = 0;

    for (size_t g = 0; g < group_cnt; ++g) {
        uint8_t b0 = input_data[s_idx++];
        uint8_t b1 = input_data[s_idx++];
        uint8_t b2 = input_data[s_idx++];

        uint16_t p0 = ((uint16_t)b0 << 4) | ( b2 & 0x0F );
        uint16_t p1 = ((uint16_t)b1 << 4) | ( b2 >> 4 );

        output_cpu[d_idx++] = p0;
        output_cpu[d_idx++] = p1;
    }
}
//raw12_to_uint16
int subads_ref(unsigned char *input1, unsigned char *input2, unsigned char *output, int img_size)
{
    for (int i = 0; i < img_size; i++) output[i] = abs(input1[i] - input2[i]);

    return 0;
}

int threshold_ref(unsigned char *input, unsigned char *output, int height, int width,
                    TPU_THRESHOLD_TYPE threshold_type, unsigned char threshold,
                    unsigned char max_value)
{
    switch (threshold_type) {
      case THRESHOLD_BINARY:
        for (int i = 0; i < width * height; i++) {
          if (input[i] > threshold)
            output[i] = max_value;
          else
            output[i] = 0;
        }
        break;
      case THRESHOLD_BINARY_INV:
        for (int i = 0; i < width * height; i++) {
          if (input[i] > threshold)
            output[i] = 0;
          else
            output[i] = max_value;
        }
        break;
      case THRESHOLD_TRUNC:
        for (int i = 0; i < width * height; i++) {
          if (input[i] > threshold)
            output[i] = threshold;
          else
            output[i] = input[i];
        }
        break;
      case THRESHOLD_TOZERO:
        for (int i = 0; i < width * height; i++) {
          if (input[i] > threshold)
            output[i] = input[i];
          else
            output[i] = 0;
        }
        break;
      case THRESHOLD_TOZERO_INV:
        for (int i = 0; i < width * height; i++) {
          if (input[i] > threshold)
            output[i] = 0;
          else
            output[i] = input[i];
        }
        break;
      default:
        break;
    }

    return 0;
}

static unsigned char blend_pixel(unsigned char left, unsigned char right, unsigned char alpha) {
    float r = (alpha * left + (255 - alpha) * right) / 255;

    return (r > 255) ? 255 : (unsigned char)r;
}

static void process_uv_plane(unsigned char *blend_uv_base, unsigned char *left_uv_base,
                               unsigned char *right_uv_base, int format, int bheight, int bwidth,
                               int uv_bstride, int lheight, int uv_lstride, int rheight, int rwidth,
                               int uv_rstride, unsigned char *wgt, TPU_BLEND_WGT_MODE wgt_mode,
                               int overlay_lx, int overlay_rx)
{
    unsigned char *wgt_uv = NULL;
    unsigned char alpha_u = 0, alpha_v = 0;
    int overlay_w = overlay_rx - overlay_lx + 1;
    int overlay_uv_w = overlay_w / 2;

    int uv_bheight = ALIGN(bheight / 2, 2);
    int uv_bwidth = ALIGN(bwidth / 2, 2);

    int uv_lheight = ALIGN(lheight / 2, 2);

    int uv_rheight = ALIGN(rheight / 2, 2);
    int uv_rwidth = ALIGN(rwidth / 2, 2);

    int buv_size = uv_bheight * uv_bstride;
    int luv_size = uv_lheight * uv_lstride;
    int ruv_size = uv_rheight * uv_rstride;

    if (wgt_mode == WGT_YUV_SHARE) {
      wgt_uv = (unsigned char *)malloc(uv_bheight * overlay_uv_w);
      for (int y = 0; y < uv_bheight; y++) {
        for (int x = 0; x < overlay_uv_w; x++) {
          int offset00 = overlay_w * y * 2 + x * 2;
          int offset01 = overlay_w * y * 2 + x * 2 + 1;
          int offset10 = overlay_w * (y * 2 + 1) + x * 2;
          int offset11 = overlay_w * (y * 2 + 1) + x * 2 + 1;
          wgt_uv[y * overlay_uv_w + x] =
              (wgt[offset00] + wgt[offset01] + wgt[offset10] + wgt[offset11]) >> 2;
        }
      }
    }

    for (int y = 0; y < uv_bheight; y++) {
      for (int x = 0; x < uv_bwidth; x++) {
        int blend_u_offet = y * uv_bstride + x;
        int blend_v_offet = buv_size + y * uv_bstride + x;

        if (format == PIXEL_FORMAT_NV12 || format == PIXEL_FORMAT_NV21) {
          blend_u_offet = y * uv_bstride + (format == PIXEL_FORMAT_NV12 ? 2 * x : 2 * x + 1);
          blend_v_offet = y * uv_bstride + (format == PIXEL_FORMAT_NV12 ? 2 * x + 1 : 2 * x);
        }

        if (x < (overlay_lx / 2)) {
          int left_u_pos = y * uv_lstride + x;
          int left_v_pos = luv_size + y * uv_lstride + x;

          if (format == PIXEL_FORMAT_NV12 || format == PIXEL_FORMAT_NV21) {
            left_u_pos = y * uv_lstride + (format == PIXEL_FORMAT_NV12 ? 2 * x : 2 * x + 1);
            left_v_pos = y * uv_lstride + (format == PIXEL_FORMAT_NV12 ? 2 * x + 1 : 2 * x);
          }

          blend_uv_base[blend_u_offet] = left_uv_base[left_u_pos];
          blend_uv_base[blend_v_offet] = left_uv_base[left_v_pos];

        } else if (x > (overlay_rx / 2)) {
          int right_x = x - overlay_lx / 2;
          if (right_x >= uv_rwidth) right_x = uv_rwidth - 1;

          int right_u_pos = y * uv_rstride + right_x;
          int right_v_pos = ruv_size + y * uv_rstride + right_x;

          if (format == PIXEL_FORMAT_NV12 || format == PIXEL_FORMAT_NV21) {
            right_u_pos =
                y * uv_rstride + (format == PIXEL_FORMAT_NV12 ? right_x * 2 : right_x * 2 + 1);
            right_v_pos =
                y * uv_rstride + (format == PIXEL_FORMAT_NV12 ? right_x * 2 + 1 : right_x * 2);
          }

          blend_uv_base[blend_u_offet] = right_uv_base[right_u_pos];
          blend_uv_base[blend_v_offet] = right_uv_base[right_v_pos];
        } else {
          if (wgt_mode == WGT_YUV_SHARE) {
            alpha_u = alpha_v = wgt_uv[y * overlay_uv_w + (x - overlay_lx / 2)];
          } else if (wgt_mode == WGT_UV_SHARE) {
            if (format == PIXEL_FORMAT_YUV_PLANAR_420) {
              alpha_u = alpha_v = wgt[bheight * overlay_w + y * overlay_uv_w + (x - overlay_lx / 2)];
            } else {
              int alpha_ux = format == PIXEL_FORMAT_NV12 ? 2 * (x - overlay_lx / 2)
                                                         : 2 * (x - overlay_lx / 2) + 1;
              int alpha_vx = format == PIXEL_FORMAT_NV12 ? 2 * (x - overlay_lx / 2) + 1
                                                         : 2 * (x - overlay_lx / 2);

              alpha_u = wgt[overlay_w * bheight + y * overlay_w + alpha_ux];
              alpha_v = wgt[overlay_w * bheight + y * overlay_w + alpha_vx];
            }
          }

          int left_u_pos = y * uv_lstride + x;
          int left_v_pos = luv_size + y * uv_lstride + x;

          if (format == PIXEL_FORMAT_NV12 || format == PIXEL_FORMAT_NV21) {
            left_u_pos = y * uv_lstride + (format == PIXEL_FORMAT_NV12 ? 2 * x : 2 * x + 1);
            left_v_pos = y * uv_lstride + (format == PIXEL_FORMAT_NV12 ? 2 * x + 1 : 2 * x);
          }

          unsigned char left_u = left_uv_base[left_u_pos];
          unsigned char left_v = left_uv_base[left_v_pos];

          int right_x = x - overlay_lx / 2;
          if (right_x >= uv_rwidth) right_x = uv_rwidth - 1;

          int right_u_pos = y * uv_rstride + right_x;
          int right_v_pos = ruv_size + y * uv_rstride + right_x;

          if (format == PIXEL_FORMAT_NV12 || format == PIXEL_FORMAT_NV21) {
            right_u_pos =
                y * uv_rstride + (format == PIXEL_FORMAT_NV12 ? right_x * 2 : right_x * 2 + 1);
            right_v_pos =
                y * uv_rstride + (format == PIXEL_FORMAT_NV12 ? right_x * 2 + 1 : right_x * 2);
          }

          unsigned char right_u = right_uv_base[right_u_pos];
          unsigned char right_v = right_uv_base[right_v_pos];

          blend_uv_base[blend_u_offet] = blend_pixel(left_u, right_u, alpha_u);
          blend_uv_base[blend_v_offet] = blend_pixel(left_v, right_v, alpha_v);
        }
      }
    }

    if (wgt_mode == WGT_YUV_SHARE) free(wgt_uv);
}

int cpu_2way_blend(int lwidth, int lheight, int *left_stride, unsigned char *left_img, int rwidth,
                     int rheight, int *right_stride, unsigned char *right_img, int bwidth,
                     int bheight, int *blend_stride, unsigned char *blend_img, int overlay_lx,
                     int overlay_rx, unsigned char *wgt, TPU_BLEND_WGT_MODE wgt_mode, int channel,
                     int format)
{
    unsigned char alpha = 0;

    if (lheight != rheight) {
      printf("left_img right img height");
    }

    int overlay_w = overlay_rx - overlay_lx + 1;
    int lstride = 0, luv_stride = 0, rstride = 0, ruv_stride = 0, bstride = 0, buv_stride = 0;

    if (left_stride != NULL) {
      lstride = left_stride[0];
      if (format == PIXEL_FORMAT_YUV_PLANAR_420 || format == PIXEL_FORMAT_NV12 ||
          format == PIXEL_FORMAT_NV21)
        luv_stride = left_stride[1];
    }

    if (right_stride != NULL) {
      rstride = right_stride[0];
      if (format == PIXEL_FORMAT_YUV_PLANAR_420 || format == PIXEL_FORMAT_NV12 ||
          format == PIXEL_FORMAT_NV21)
        ruv_stride = right_stride[1];
    }

    if (blend_stride != NULL) {
      bstride = blend_stride[0];
      if (format == PIXEL_FORMAT_YUV_PLANAR_420 || format == PIXEL_FORMAT_NV12 ||
          format == PIXEL_FORMAT_NV21)
        buv_stride = blend_stride[1];
    }

    if (format == PIXEL_FORMAT_YUV_PLANAR_420 || format == PIXEL_FORMAT_NV12 ||
        format == PIXEL_FORMAT_NV21) {
      // Y channel
      for (int y = 0; y < bheight; y++) {
        for (int x = 0; x < bwidth; x++) {
          if (x < overlay_lx) {
            blend_img[y * bstride + x] = left_img[y * lstride + x];
          } else if (x > overlay_rx) {
            int right_x = x - overlay_lx;
            if (right_x >= rwidth) right_x = rwidth - 1;

            blend_img[y * bstride + x] = right_img[y * rstride + right_x];
          } else {
            alpha = wgt[y * overlay_w + (x - overlay_lx)];
            unsigned char left_p = left_img[y * lstride + x];

            int right_x = x - overlay_lx;
            if (right_x < 0) right_x = 0;
            if (right_x >= rwidth) right_x = rwidth - 1;
            unsigned char right_p = right_img[y * rstride + right_x];

            blend_img[y * bstride + x] = blend_pixel(left_p, right_p, alpha);
          }
        }
      }

      process_uv_plane(blend_img + bheight * bstride, left_img + lheight * lstride,
                       right_img + rheight * rstride, format, bheight, bwidth, buv_stride, lheight,
                       luv_stride, rheight, rwidth, ruv_stride, wgt, wgt_mode, overlay_lx,
                       overlay_rx);
    } else {
      for (int c = 0; c < channel; c++) {
        for (int y = 0; y < bheight; y++) {
          for (int x = 0; x < bwidth; x++) {
            if (x < overlay_lx) {
              blend_img[c * bheight * bstride + y * bstride + x] =
                  left_img[c * lheight * lstride + y * lstride + x];
            } else if (x > overlay_rx) {
              int right_x = x - overlay_lx;
              if (right_x >= rwidth) right_x = rwidth - 1;
              blend_img[c * bheight * bstride + y * bstride + x] =
                  right_img[c * rheight * rstride + y * rstride + right_x];
            } else {
              alpha = wgt[y * overlay_w + (x - overlay_lx)];
              unsigned char left_pixel = left_img[c * lheight * lstride + y * lstride + x];
              unsigned char right_pixel =
                  right_img[c * rheight * rstride + y * rstride + (x - overlay_lx)];

              blend_img[c * bheight * bstride + y * bstride + x] =
                  blend_pixel(left_pixel, right_pixel, alpha);
            }
          }
        }
      }
    }

    return 0;
}

void grayscale_dilate(const unsigned char* src,
                      unsigned char* dst,
                      int width, int height,
                      int w_stride, int ksize)
{
    const int radius = ksize / 2;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            unsigned char max_val = 0;

            for (int ky = -radius; ky <= radius; ky++) {
                for (int kx = -radius; kx <= radius; kx++) {
                    int ny = y + ky;
                    int nx = x + kx;

                    if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
                        if (src[ny * w_stride + nx] > max_val) {
                            max_val = src[ny * w_stride + nx];
                        }
                    }
                }
            }
            dst[y * w_stride + x] = max_val;
        }
    }
}

void grayscale_erode(const unsigned char* src,
                      unsigned char* dst,
                      int width, int height,
                      int w_stride, int ksize)
{
    const int radius = ksize / 2;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            unsigned char min_val = UCHAR_MAX;

            for (int ky = -radius; ky <= radius; ky++) {
                for (int kx = -radius; kx <= radius; kx++) {
                    int ny = y + ky;
                    int nx = x + kx;

                    if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
                        if (src[ny * w_stride + nx] < min_val) {
                            min_val = src[ny * w_stride + nx];
                        }
                    }
                }
            }
            dst[y * w_stride + x] = min_val;
        }
    }
}