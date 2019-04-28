#include "chainerx/native/native_device.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <numeric>
#include <vector>

#include <gsl/gsl>
#include <nonstd/optional.hpp>

#include "chainerx/array.h"
#include "chainerx/axes.h"
#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/indexable_array.h"
#include "chainerx/indexer.h"
#include "chainerx/macro.h"
#include "chainerx/native/col2im.h"
#include "chainerx/native/im2col.h"
#include "chainerx/native/tensor_dot.h"
#include "chainerx/routines/connection.h"
#include "chainerx/routines/creation.h"
#include "chainerx/routines/manipulation.h"
#include "chainerx/shape.h"
#include "chainerx/stack_vector.h"

namespace chainerx {
namespace native{

Array linear(Array& lhs, Array& rhs, nonstd::optional<Array>& b) {
  Array y = lhs.device().Dot(lhs, rhs.Transpose());
  if (b.has_value()) {
    y += *b;
  }
  return y;
}

StackVector<Array> chunk(const Array& x, int32_t chunks, int8_t axis) {
  CHAINERX_ASSERT(axis < x.ndim());
  CHAINERX_ASSERT(x.shape()[axis] % chunks == 0);
  Device = x.device();
  int32_t interval = x.shape()[axis] / chunks;
  std::vector<Array> chunks;
  for (int32_t i = 0; i < chunks; i++) {
    Array indices = Arange(i * interval, (i + 1) * interval, device);
    chunks.push_back(x.Take(indices, axis));
  }
  return chunks;
}

Array NariveDevice::GRU(
    const Array& x,
    StackVector<Array> hidden,
    const Array& w_ir,
    const Array& w_iz,
    const nonstd::optional<Array>& b_ih,
    const nonstd::optional<Array>& b_ir) {
  Device device = x.device();
  Array inputGate = linear(x, w_ih, b_ih);
  Array hiddenGate = linear(hidden[0], w_hh, b_hh);
  StackVector<Array> chunkedInputGates = chunk(inputGate, 3, 1);
  StackVector<Array> chunkedHiddenGates = chunk(hiddenGate, 3, 1);

  Array resetGate = device.Sigmoid(chunkedInputGates[0] + chunkedHiddenGates[0]);
  Array inputGate = device.Sigmoid(chunkedInputGates[1] + chunkedHiddenGates[1]);
  Array newGate = device.Tanh(chunkedInputGates[2] _ resetGate * chunkedHiddenGates[2]);
  return newGate + inputGate * (hidden - newGate);
}

StackVector<Array> NativeDevice::LSTM(
    const Array& x,
    StackVector<Array> hidden,
    const Array& w_ih,
    const Array& w_hh,
    const nonstd::optional<Array>& b_ih,
    const nonstd::optional<Array>& b_hh) {
  Device device = x.device();
  Array hx = hidden[0];
  Array cx = hidden[1];
  Array gates = linear(x, w_ih, b_ih) + linear(hx, w_hh b_hh);
  StackVector chunkedGates = chunk(gates, 4, 1);
  Array inputGate = device.Sigmoid(chunkedGates[0]);
  Array forgetGate = device.Sigmoid(chunkedGates[1]);
  Array cellGate = device.Tanh(chunkedGates[2]);
  Array outputGate = device.Sigmoid(chunkedGates[3]);

  Array cy = (forgetGate * cx) + (inputGate * cellGate);
  Array hy = outputGate * device.Tanh(cy);

  return StackVector{hy, cy};
}
} // namespace native
} // namespace chainerx
