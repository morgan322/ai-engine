/*************************************************************************
 * Copyright (C) [2022] by Cambricon, Inc. All rights reserved
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *************************************************************************/

#ifndef AI_ENCODE_IMPL_HPP_
#define AI_ENCODE_IMPL_HPP_

#include "codec/ai_encoder.h"

namespace ai {
namespace node {

class IEncoder {
public:
  virtual ~IEncoder() {}
  virtual int Create(AIVencCreateParams *params) = 0;
  virtual int Destroy() = 0;
  virtual int SendFrame(AIBufSurface *surf, int timeout_ms) = 0;
};

IEncoder *CreateEncoder();

} // namespace node
} // namespace ai

#endif // AI_ENCODE_IMPL_HPP_
