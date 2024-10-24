#ifndef INFER_SERVER_CORE_REQUEST_CTRL_H_
#define INFER_SERVER_CORE_REQUEST_CTRL_H_

#include <glog/logging.h>

#include <cassert>
#include <chrono>
#include <functional>
#include <future>
#include <map>
#include <mutex>
#include <string>
#include <utility>

#include "inference/infer_server.h"


namespace infer_server {

class RequestControl {
 public:
  using ResponseFunc = std::function<void(Status, PackagePtr)>;
  using NotifyFunc = std::function<void(const RequestControl*)>;

  RequestControl(ResponseFunc&& response, NotifyFunc&& done_notifier, const std::string& tag, int64_t request_id,
                 uint32_t data_num) noexcept
      : output_(new Package),
        response_(std::forward<ResponseFunc>(response)),
        done_notifier_(std::forward<NotifyFunc>(done_notifier)),
        tag_(tag),
        request_id_(request_id),
        data_num_(data_num),
        wait_num_(data_num),
        process_finished_(data_num ? false : true) {
    output_->data.resize(data_num);
    assert(response_);
    assert(done_notifier_);
  }

  ~RequestControl() {
    std::lock_guard<std::mutex> lk(done_mutex_);
    response_done_flag_.set_value();
  }

  /* ---------------------------- Observer --------------------------------*/
  const std::string& Tag() const noexcept { return tag_; }
  int64_t RequestId() const noexcept { return request_id_; }
  uint32_t DataNum() const noexcept { return data_num_; }

  bool IsSuccess() const noexcept { return status_.load() == Status::SUCCESS; }
  bool IsDiscarded() const noexcept { return is_discarded_.load(); }
  bool IsProcessFinished() const noexcept { return process_finished_.load(); }
  /* -------------------------- Observer END ------------------------------*/

#ifdef CNIS_RECORD_PERF
  void BeginRecord() noexcept { start_time_ = std::chrono::steady_clock::now(); }

  float EndRecord() noexcept {
    // record request latency
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<float, std::milli> dura = end - start_time_;
    float latency = dura.count();
    output_->perf["RequestLatency"] = latency;
    return latency;
  }

  // invoked only before response
  const std::map<std::string, float>& Performance() const noexcept { return output_->perf; }
#endif

  void Response() noexcept {
    output_->tag = tag_;
    response_(status_.load(), std::move(output_));
    VLOG(4) << "[EasyDK InferServer] [RequestControl] Response end, request id: " << request_id_;
  }

  std::future<void> ResponseDonePromise() noexcept { return response_done_flag_.get_future(); }

  void Discard() noexcept { is_discarded_.store(true); }

  void ProcessFailed(Status status) noexcept { ProcessDone(status, nullptr, 0, {}); }

  // process on one piece of data done
  void ProcessDone(Status status, InferDataPtr output, uint32_t index, std::map<std::string, float>&& perf) noexcept;

 private:
  RequestControl() = delete;
  PackagePtr output_;
  ResponseFunc response_;
  NotifyFunc done_notifier_;
  std::string tag_;
  std::mutex done_mutex_;
  std::promise<void> response_done_flag_;
  int64_t request_id_;
  uint32_t data_num_;
  uint32_t wait_num_;
  std::atomic<Status> status_{Status::SUCCESS};
  std::atomic<bool> is_discarded_{false};
  std::atomic<bool> process_finished_{false};
#ifdef CNIS_RECORD_PERF
  std::chrono::time_point<std::chrono::steady_clock> start_time_;
#endif
};

}  // namespace infer_server

#endif  // INFER_SERVER_CORE_REQUEST_CTRL_H_
