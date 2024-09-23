#ifndef CODEC_DECODE_HPP_
#define CODEC_DECODE_HPP_

#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "glog/logging.h"

#include "../basic/basic_module.hpp"
#include "ffmpeg_demuxer.h"
#include "framerate_contrller.h"

namespace ai {
namespace codec {
class Decode : public ai::basic::BasicModule {
public:
  Decode(std::string name, int parallelism, int device_id,
               std::string filename, int stream_id, int frame_rate = 30)
      : BasicModule(name, parallelism) {
    filename_ = filename;
    stream_id_ = stream_id;
    dev_id_ = device_id;
    if (frame_rate > 0) {
      frame_rate_ = frame_rate;
    }
  }

  ~Decode();

  int Open() override;

  int Close() override;

  int Process(std::shared_ptr<ai::basic::Frame> frame) override;

  static int GetBufSurface_(CnedkBufSurface **surf, int width, int height,
                            CnedkBufSurfaceColorFormat fmt, int timeout_ms,
                            void *userdata) {
    Decode *thiz = reinterpret_cast<Decode *>(userdata);
    return thiz->GetBufSurface(surf, width, height, fmt, timeout_ms);
  }
  static int OnFrame_(CnedkBufSurface *surf, void *userdata) {
    Decode *thiz = reinterpret_cast<Decode *>(userdata);
    return thiz->OnFrame(surf);
  }
  static int OnEos_(void *userdata) {
    Decode *thiz = reinterpret_cast<Decode *>(userdata);
    return thiz->OnEos();
  }
  static int OnError_(int errcode, void *userdata) {
    Decode *thiz = reinterpret_cast<Decode *>(userdata);
    return thiz->OnError(errcode);
  }

private:
  int CreateSurfacePool(void **surf_pool, int width, int height);
  int GetBufSurface(CnedkBufSurface **surf, int width, int height,
                    CnedkBufSurfaceColorFormat fmt, int timeout_ms);
  int OnFrame(CnedkBufSurface *surf);
  int OnEos();
  int OnError(int errcode);

private:
  int stream_id_;
  std::string filename_;
  uint64_t frame_count_ = 0;

private:
  CnedkVdecCreateParams params_;
  std::unique_ptr<FFmpegDemuxer> demuxer_;
  std::unique_ptr<FrController> fr_controller_;
  int width_;
  int height_;
  void *vdec_ = nullptr;
  bool eos_send_ = false;
  void *surf_pool_ = nullptr;
  int dev_id_ = 0;
  int frame_rate_ = 30;

  uint8_t *data_buffer_ = nullptr;
};
} // namespace codec
} // namespace ai