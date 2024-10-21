
#ifndef BASIC_MODULE_HPP_
#define BASIC_MODULE_HPP_

#include "frame.h"

#include <functional>
#include <memory>
#include <string>

namespace ai
{
  namespace module
  {
    class BasicModule
    {
    public:
      BasicModule(std::string name, int parallelism)
          : module_name_(name), parallelism_(parallelism) {}

      ~BasicModule() = default;

      void
      SetProcessDoneCallback(std::function<int(std::shared_ptr<ai::module::Frame>)> callback)
      {
        callback_ = callback;
      }

      virtual int Transmit(std::shared_ptr<ai::module::Frame> frame) final
      { // NOLINT
        if (callback_)
          return callback_(frame);
        return 0;
      }

      int GetParallelism() { return parallelism_; }

      std::string GetModuleName() { return module_name_; }

    public:
      virtual int Open() = 0;
      virtual int Process(std::shared_ptr<ai::module::Frame> frame) = 0;
      virtual int Close() = 0;

    private:
      std::string module_name_ = "";
      int parallelism_ = 1;
      std::function<int(std::shared_ptr<ai::module::Frame>)> callback_;
      std::shared_ptr<BasicModule> next_module_ = nullptr;
    };

  } // namespace basic
} // namespace ai
#endif // BASIC_MODULE_HPP_