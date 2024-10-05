#ifndef SAMPLE_ENCODE_HPP_
#define SAMPLE_ENCODE_HPP_

#include <condition_variable>
#include <map>
#include <memory>
#include <string>

#include "easy_module.hpp"

#include "cnedk_encode.h"
#include "encode_handler_mlu.hpp"

namespace ai {
namespace codec {

class Encode : public ai::basic::BasicModule {
public:
  explicit Encode(std::string name, int parallelism, int device_id,
                  std::string filename)
      : BasicModule(name, parallelism) {
    device_id_ = device_id;
    filename_ = filename;
  }

  ~Encode();

  int Open() override;

  int Close() override;

  int Process(std::shared_ptr<ai::basic::Frame> frame) override;

private:
  int device_id_;
  VEncHandlerParam param_;
  std::mutex venc_mutex_;
  std::string filename_;
  std::map<int, std::shared_ptr<VencMluHandler>> ivenc_;
};

} // namespace codec
} // namespace ai

#endif
