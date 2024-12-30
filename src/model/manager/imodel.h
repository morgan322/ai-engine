#ifndef INFER_SERVER_MM_HELPER_H_
#define INFER_SERVER_MM_HELPER_H_

#include <memory>
#include <iostream>

namespace infer_server {


struct ai_model
{
  
};





class InferDeleter {
  template <typename T>
  void operator()(T *obj) const {
    if (obj) {
      obj->Destroy();
    }
  }
};
template <typename T>
using mm_unique_ptr = std::unique_ptr<T, InferDeleter>;


using MContext = IContext;
using MModel = IModel;
using MEngine = IEngine;
using MTensor = IRTTensor;

}  // namespace infer_server

#endif  // INFER_SERVER_MM_HELPER_H_
