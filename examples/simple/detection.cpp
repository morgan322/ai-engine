#include <gflags/gflags.h>
#include <glog/logging.h>
#include <unistd.h>

#include <csignal>
#include <future>
#include <iostream>
#include <memory>
#include <utility>

#include "codec/video_decoder.h"
#include "runner/detection_runner.h"
#include "runner/stream_runner.h"

// #include "codec/opencv_decoder.hpp"

// #include "AI_vin_capture.h"
// #include "AI_vout_display.h"
// #include "AI_platform.h"


DEFINE_bool(show, false, "show image");
DEFINE_bool(save_video, true, "save output to local video file");
DEFINE_int32(repeat_time, 0, "process repeat time");
DEFINE_string(data_path, "", "video path");
DEFINE_string(model_path, "", "infer offline model path");
DEFINE_string(label_path, "", "label path");
DEFINE_int32(wait_time, 0, "time of one test case");
DEFINE_int32(dev_id, 0, "run sample on which device");
DEFINE_string(decode_type, "ffmpeg", "decode type, choose from camera/ffmpeg/opencv.");
// DEFINE_bool(enable_vin, false, "enable_vin");  // not support vin enable
// DEFINE_bool(enable_vout, false, "enable_vout");  // not support vout enable
DEFINE_int32(codec_id_start, 0, "vdec/venc first id, for CE3226 only");

std::shared_ptr<StreamRunner> g_runner;
bool g_exit = false;

void HandleSignal(int sig) {
  g_runner->Stop();
  g_exit = true;
  LOG(INFO) << "[AI Simple] [Detection] Got INT signal, ready to exit!";
}

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  FLAGS_stderrthreshold = google::INFO;
  FLAGS_colorlogtostderr = true;

  // check params
  CHECK(FLAGS_data_path.size() != 0u) << "[AI Simple] [Detection] data path is empty";       // NOLINT
  CHECK(FLAGS_model_path.size() != 0u) << "[AI Simple] [Detection] model path is empty";     // NOLINT
  CHECK(FLAGS_label_path.size() != 0u) << "[AI Simple] [Detection] label path is empty";     // NOLINT
  CHECK(FLAGS_wait_time >= 0) << "[AI Simple] [Detection] wait time should be >= 0";         // NOLINT
  CHECK(FLAGS_repeat_time >= 0) << "[AI Simple] [Detection] repeat time should be >= 0";     // NOLINT
  CHECK(FLAGS_dev_id >= 0) << "[AI Simple] [Detection] device id should be >= 0";            // NOLINT
  CHECK(FLAGS_codec_id_start >= 0) "[AI Simple] [Detection] codec start id should be >= 0";  // NOLINT

  AISensorParams sensor_params[4];
  memset(sensor_params, 0, sizeof(AISensorParams) * 4);
  AIVoutParams vout_params;
  memset(&vout_params, 0, sizeof(AIVoutParams));

  AIPlatformConfig config;
  memset(&config, 0, sizeof(config));
  if (FLAGS_codec_id_start) {
    config.codec_id_start = FLAGS_codec_id_start;
  }
  // if (FLAGS_enable_vout) {
  //   config.vout_params = &vout_params;
  //   vout_params.max_input_width = 1920;
  //   vout_params.max_input_height = 1080;
  //   vout_params.input_format = 0;  // not used at the moment
  // }
  // if (FLAGS_enable_vin) {
  //   config.sensor_num = 1;
  //   config.sensor_params = sensor_params;
  //   sensor_params[0].sensor_type = 6;
  //   sensor_params[0].mipi_dev = 1;
  //   sensor_params[0].bus_id = 0;
  //   sensor_params[0].sns_clk_id = 1;
  //   sensor_params[0].out_width = 1920;
  //   sensor_params[0].out_height = 1080;
  //   sensor_params[0].output_format = 0;  // not used at the moment
  // }

  // if (AIPlatformInit(&config) < 0) {
  //   LOG(ERROR) << "[AI Simple] [Detection] Init platform failed";
  //   return -1;
  // }

  VideoDecoder::DecoderType decode_type = VideoDecoder::CAMERA;
  if (FLAGS_decode_type == "ffmpeg" || FLAGS_decode_type == "FFmpeg") {
    decode_type = VideoDecoder::FFMPEG;
  } else if (FLAGS_decode_type == "OpenCV" || FLAGS_decode_type == "opencv") {
    decode_type = VideoDecoder::OPENCV;
  }
  try {
    g_runner = std::make_shared<DetectionRunner>(decode_type, FLAGS_dev_id, FLAGS_model_path, FLAGS_label_path,
                                                 FLAGS_data_path, FLAGS_show, FLAGS_save_video);
  } catch (...) {
    LOG(ERROR) << "[AI Simple] [Detection] Create stream runner failed";
    return -1;
  }

  std::future<bool> process_loop_return = std::async(std::launch::async, &StreamRunner::RunLoop, g_runner.get());

  if (0 < FLAGS_wait_time) {
    alarm(FLAGS_wait_time);
  }

  signal(SIGALRM, HandleSignal);

  g_runner->DemuxLoop(FLAGS_repeat_time);

  process_loop_return.wait();
  g_runner.reset();

  if (!process_loop_return.get()) {
    return 1;
  }

  LOG(INFO) << "[AI Simple] [Detection] Run SUCCEED!!!";
  google::ShutdownGoogleLogging();
  return 0;
}
