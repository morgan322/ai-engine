#ifndef AI_CODEC_VIDEO_DECODER_H_
#define AI_CODEC_VIDEO_DECODER_H_

#ifdef __cplusplus
extern "C" {
#endif
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/imgutils.h>
#if LIBAVFORMAT_VERSION_INT == FFMPEG_VERSION_4_2_2
#include <libavutil/hwcontext.h>
#endif
#ifdef __cplusplus
}
#endif

#include <utility>

#include "buf_surface.h"

#include "video_parser.h"

class IDecodeEventHandle {
 public:
  virtual void OnDecodeFrame(AIBufSurface* surf) = 0;
  virtual void OnDecodeEos() = 0;
};

class StreamRunner;
class VideoDecoder;

class VideoDecoderImpl {
 public:
  explicit VideoDecoderImpl(VideoDecoder* interface, IDecodeEventHandle* handle, int device_id)
      : interface_(interface), handle_(handle), device_id_(device_id) {}
  virtual ~VideoDecoderImpl() = default;
  virtual bool Init() = 0;
  virtual bool FeedPacket(const AVPacket* pkt) = 0;
  virtual void FeedEos() = 0;
  virtual void ReleaseFrame(AIBufSurface* frsurfame) = 0;
  // virtual bool CopyFrameD2H(void *dst, const CnBufSurface& surf) = 0;
  virtual void Destroy() = 0;

 protected:
  VideoDecoder* interface_;
  IDecodeEventHandle* handle_;
  int device_id_;
};

class VideoDecoder final : public IDemuxEventHandle {
 public:
  enum DecoderType {
    CAMERA,
    FFMPEG,
    OPENCV
  };

  VideoDecoder(StreamRunner* runner, DecoderType type, int device_id);
  bool OnParseInfo(const VideoInfo& info) override;
  bool OnPacket(const AVPacket* packet) override;
  void OnEos() override;
  bool Running() override;
  VideoInfo& GetVideoInfo() { return info_; }
  // bool CopyFrameD2H(void *dst, AIBufSurface* surf) { return true; }
  // void ReleaseFrame(AIBufSurface* surf) { AIBufSurfaceDestroy(surf); }
  void Destroy();
  ~VideoDecoder();

 private:
  VideoInfo info_;
  const StreamRunner* runner_{nullptr};
  VideoDecoderImpl* impl_{nullptr};
  bool send_eos_{false};
};

#endif  // AI_SAMPLES_VIDEO_PARSER_H_
