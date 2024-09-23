#ifndef AI_DECODE_H_
#define AI_DECODE_H_

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
  AI_VDEC_TYPE_INVALID,
  /** Specifies H264 codec type */
  AI_VDEC_TYPE_H264,
  /** Specifies H265/HEVC codec type */
  AI_VDEC_TYPE_H265,
  /** Specifies JPEG codec type */
  AI_VDEC_TYPE_JPEG,
  /** Specifies the number of codec types */
  AI_VDEC_TYPE_NUM
} AIVdecType;

/**
 * Holds the parameters for creating video decoder.
 */
typedef struct AIVdecCreateParams {
  /** The id of device where the decoder will be created. */
  int device_id;
  /** The video codec type. */
  AIVdecType type;
  /** The max width of the frame that the decoder could handle. */
  uint32_t max_width;
  /** The max height of the frame that the decoder could handle. */
  uint32_t max_height;
  /** The number of frame buffers that the decoder will allocated. Only valid on CE3226 platform */
  uint32_t frame_buf_num;
  /** The color format of the frame after decoding. */
  AIBufSurfaceColorFormat color_format;

  // When a decoded picture got, the below steps will be performed
  //  (1)  GetBufSurf
  //  (2)  decoded picture to buf_surf (csc/deepcopy)
  //  (3)  OnFrame
  /** The OnFrame callback function.*/
  int (*OnFrame)(AIBufSurface *surf, void *userdata);
  /** The OnEos callback function. */
  int (*OnEos)(void *userdata);
  /** The OnError callback function. */
  int (*OnError)(int errcode, void *userdata);
  /** The GetBufSurf callback function. */
  int (*GetBufSurf)(AIBufSurface **surf,
                  int width, int height, AIBufSurfaceColorFormat fmt,
                  int timeout_ms, void *userdata) = 0;
  /** The timeout in milliseconds. */
  int surf_timeout_ms;
  /** The user data. */
  void *userdata;
} AIVdecCreateParams;

/**
 * Holds the video stream.
 */
typedef struct AIVdecStream {
  /** The data of the video stream. */
  uint8_t *bits;
  /** The length of the video stream. */
  uint32_t len;
  /** The flags of the video stream. Not valid on CE3226 platform */
  uint32_t flags;
  /** The presentation timestamp of the video stream. */
  uint64_t pts;
} AIVdecStream;

/**
 * @brief Creates a video decoder with the given parameters.
 *
 * @param[out] vdec A pointer points to the pointer of a video decoder.
 * @param[in] params The parameters for creating video decoder.
 *
 * @return Returns 0 if this function has run successfully. Otherwise returns -1.
 */
int AIVdecCreate(void **vdec, AIVdecCreateParams *params);
/**
 * @brief Destroys a video decoder.
 *
 * @param[in] vdec A pointer of a video decoder.
 *
 * @return Returns 0 if this function has run successfully. Otherwise returns -1.
 */
int AIVdecDestroy(void *vdec);
/**
 * @brief Sends video stream to a video decoder.
 *
 * @param[in] vdec A pointer of a video decoder.
 * @param[in] stream The video stream.
 * @param[in] timeout_ms The timeout in milliseconds.
 *
 * @return Returns 0 if this function has run successfully. Otherwise returns -1.
 */
int AIVdecSendStream(void *vdec, const AIVdecStream *stream, int timeout_ms);

#ifdef __cplusplus
};
#endif

#endif  // AI_DECODE_H_
