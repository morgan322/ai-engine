#ifndef EDK_SAMPLES_STREAM_APP_DETECTION_RUNNER_H_
#define EDK_SAMPLES_STREAM_APP_DETECTION_RUNNER_H_

#include <memory>
#include <string>
#include <vector>

#include "inference/processor.h"
#include "inference/infer_server.h"
#include "utils/osd.h"
#include "runner/stream_runner.h"
#include "module/frame.h"
#include "platform/ai_platform.h"

#include "model/class_detector.h"
#include "class_timer.hpp"

struct DetectionFrame {
  AIBufSurface* surf;
  std::vector<ai::module::DetectObject> objs;
};

class DetectionRunner : public StreamRunner, public infer_server::IPreproc, public infer_server::IPostproc {
 public:
  DetectionRunner(const VideoDecoder::DecoderType& decode_type, int device_id,
                  const std::string& model_path, const std::string& label_path,
                  const std::string& data_path, bool show, bool save_video);
  ~DetectionRunner();

  void Process(AIBufSurface* surf) override;

 private:
  int OnTensorParams(const infer_server::CnPreprocTensorParams *params) override;
  int OnPreproc(ai::BufSurfWrapperPtr src, ai::BufSurfWrapperPtr dst,
                const std::vector<AITransformRect>& src_rects) override;
  int OnPostproc(const std::vector<infer_server::InferData*>& data_vec, const infer_server::ModelIO& model_output,
                 const infer_server::ModelInfo* model_info) override;

 private:
  cv::Mat ConvertToMatAndReleaseBuf(AIBufSurface* surf);
  std::unique_ptr<infer_server::InferServer> infer_server_;
  infer_server::Session_t session_;
  AIOsd osd_;
  std::unique_ptr<cv::VideoWriter> video_writer_{nullptr};

  bool show_;
  bool save_video_;
  float threshold_ = 0.6;
  infer_server::CnPreprocTensorParams params_;
  std::unique_ptr<Detector> detector;
};

#endif  // EDK_SAMPLES_STREAM_APP_DETECTION_RUNNER_H_

