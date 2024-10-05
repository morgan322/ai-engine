#ifndef AI_ENCODE_H_
#define AI_ENCODE_H_

#include <stdint.h>
#include <stdbool.h>
#include "buf_surface.h"

#ifdef __cplusplus
extern "C" {
#endif
/**
 * Specifies codec types.
 */
typedef enum {
  /** Specifies an invalid video codec type. */
  AI_VENC_TYPE_INVALID,
  /** Specifies H264 codec type */
  AI_VENC_TYPE_H264,
  /** Specifies H265/HEVC codec type */
  AI_VENC_TYPE_H265,
  /** Specifies JPEG codec type */
  AI_VENC_TYPE_JPEG,
  /** Specifies the number of codec types */
  AI_VENC_TYPE_NUM
} AIVencType;
/**
 * Specifies package types.
 */
typedef enum {
  /** Specifies sps package type. */
  AI_VENC_PACKAGE_TYPE_SPS = 0,
  /** Specifies pps package type. */
  AI_VENC_PACKAGE_TYPE_PPS,
  /** Specifies key frame package type. */
  AI_VENC_PACKAGE_TYPE_KEY_FRAME,
  /** Specifies frame package type. */
  AI_VENC_PACKAGE_TYPE_FRAME,
  /** Specifies sps and pps package type. */
  AI_VENC_PACKAGE_TYPE_SPS_PPS,
  /** Specifies the number of package types */
  AI_VENC_PACKAGE_TYPE_NUM,
} AIVencPakageType;

/**
 * Holds the video frame.
 */
typedef struct AIVEncFrameBits {
  /** The data of the video frame. */
  unsigned char *bits;
  /** The length of the video frame. */
  int len;
  /** The presentation timestamp of the video frame. */
  uint64_t pts;
  /** The package type of the video frame. */
  AIVencPakageType pkt_type;  // nal-type
} AIVEncFrameBits;

/**
 * Holds the parameters for creating video encoder.
 */
typedef struct AIVencCreateParams {
  /** The id of device where the encoder will be created. */
  int device_id;
  /** The number of input frame buffers that the encoder will allocated. */
  int input_buf_num;
  /** The gop size. Only valid when encoding videos */
  int gop_size;
  /** The frame rate. Only valid when encoding videos */
  double frame_rate;
  /** The color format of the input frame sent to encoder. */
  AIBufSurfaceColorFormat color_format;
  /** The codec type of the encoder. */
  AIVencType type;
  /** The width of the input frame sent to encoder. */
  uint32_t width;
  /** The height of the input frame sent to encoder. */
  uint32_t height;
  /** The bit rate. Only valid when encoding videos */
  uint32_t bitrate;
  /** Not used */
  uint32_t key_interval;
  /** Jpeg quality. The range is [1, 100]. Higher the value higher the Jpeg quality. */
  uint32_t jpeg_quality;
  /** The OnFrameBits callback function.*/
  int (*OnFrameBits)(AIVEncFrameBits *framebits, void *userdata);
  /** The OnEos callback function. */
  int (*OnEos)(void *userdata);
  /** The OnError callback function. */
  int (*OnError)(int errcode, void *userdata);
  /** The user data. */
  void *userdata;
} AIVencCreateParams;

/**
 * @brief Creates a video encoder with the given parameters.
 *
 * @param[out] venc A pointer points to the pointer of a video encoder.
 * @param[in] params The parameters for creating video encoder.
 *
 * @return Returns 0 if this function has run successfully. Otherwise returns -1.
 */
int AIVencCreate(void **venc, AIVencCreateParams *params);
/**
 * @brief Destroys a video encoder.
 *
 * @param[in] venc A pointer of a video encoder.
 *
 * @return Returns 0 if this function has run successfully. Otherwise returns -1.
 */
int AIVencDestroy(void *venc);
/**
 * @brief Sends video frame to a video encoder.
 *
 * @param[in] venc A pointer of a video encoder.
 * @param[in] surf The video frame.
 * @param[in] timeout_ms The timeout in milliseconds.
 *
 * @return Returns 0 if this function has run successfully. Otherwise returns -1.
 */
int AIVencSendFrame(void *venc, AIBufSurface *surf, int timeout_ms);

#ifdef __cplusplus
};
#endif

#endif  // AI_ENCODE_H_
