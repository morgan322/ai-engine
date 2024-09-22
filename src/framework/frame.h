#ifndef FRAME_HPP_
#define FRAME_HPP_

#include <map>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace ai {
namespace fw {
struct BoundingBox {
  float x = 0.0;
  float y = 0.0;
  float w = 0.0;
  float h = 0.0;
};

struct DetectObject {
  float score;
  int label;
  BoundingBox bbox;
  std::string track_id;
  std::map<std::string, std::string>
      attributes; // add info into bbox, secondary infer classfication
};

class Frame {
public:
  int stream_id;
  uint64_t frame_idx;
  bool is_eos;
  std::vector<DetectObject> objs;
  std::string track_id;
  std::map<std::string, std::string>
      attributes; // add info into bbox, secondary infer classfication
  cv::Mat frame;
};
} // namespace fw

} // namespace ai

#endif