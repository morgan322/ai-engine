#ifndef CODEC_FR_CONTROLLER_
#define CODEC_FR_CONTROLLER_

#include <chrono>
#include <thread>
#include <iostream>
/***********************************************************************
 * @brief FrController is used to control the frequency of sending data.
 ***********************************************************************/

class FrController
{
public:
  explicit FrController(uint32_t frame_rate) : frame_rate_(frame_rate) {}
  void Start() { start_ = std::chrono::steady_clock::now(); }
  void Control()
  {
    if (0 == frame_rate_)
      return;
    double delay = 1000.0 / frame_rate_;
    end_ = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> diff = end_ - start_;
    auto gap = delay - diff.count() - time_gap_;
    if (gap > 0)
    {
      std::chrono::duration<double, std::milli> dura(gap);
      std::this_thread::sleep_for(dura);
      time_gap_ = 0;
    }
    else
    {
      time_gap_ = -gap;
    }
    Start();
  }
  inline uint32_t GetFrameRate() const { return frame_rate_; }
  inline void SetFrameRate(uint32_t frame_rate) { frame_rate_ = frame_rate; }

private:
  uint32_t frame_rate_ = 0;
  double time_gap_ = 0;
  std::chrono::time_point<std::chrono::steady_clock> start_, end_;
}; // class FrController

#endif
