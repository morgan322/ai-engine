#ifndef FRAME_H_
#define FRAME_H_

#include <map>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace ai {
namespace module {
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
  cv::Mat img;
};

// Define pixel format enumeration
enum class PixelFmt {
    NV12,  // NV12 pixel format
    // Other pixel formats can be added here
};

// CnFrame structure definition
struct ai_buffer {
    uint64_t pts;              // Presentation timestamp
    int width;                 // Frame width
    int height;                // Frame height
    PixelFmt pformat;         // Pixel format
    uint32_t n_planes;         // Number of planes (e.g., 2 for NV12)
    uint32_t strides[2];       // Strides for each plane
    uint32_t frame_size;       // Total frame size (in bytes)
    uint8_t* ptrs[2];          // Pointers to each plane data
    uint32_t device_id;        // Device identifier for the frame
};

} // namespace module
} // namespace ai
#endif