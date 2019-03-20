#include "chainerx/native/native_device.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <vector>

#include <gsl/gsl>
#include <nonstd/optional.hpp>

#include "chainerx/array.h"
#include "chainerx/axes.h"
#include "chainerx/device.h"
#include "chainerx/indexable_array.h"
#include "chainerx/indexer.h"
#include "chainerx/macro.h"
#include "chainerx/routines/connection.h"
#include "chainerx/routines/creation.h"
#include "chainerx/routines/manipulation.h"
#include "chainerx/shape.h"
#include "chainerx/stack_vector.h"

namespace chainerx {
namespace native {

StackVector<Array> extractGates(const Array& x) {
  const Array r = x.Reshape({x.Shape[0], x.Shape[1] / 4, 4}).Transpose({2, 0, 1});
  return StackVector<Array>{r[0], r[1], r[2], r[3]};
}

StackVector<Array> updateForward(const StackVector<Array> states) {
  // states are a, i, f, o
  StackVector<Array> ret = StackVector<Array>(states.size());
  for (size_t i = 0; i < states.size(); i ++) {
    if (i == 0) {
      ret[i] = states[i].device().Tanh(states[i]);
    } else {
      ret[i] = states[i].device().Sigmoid(states[i]);
    }
  }
  return ret;
}

StackVector<Array> NativeDevice::LSTM(
    const Array& x,
    const Array& c, /* Cell state */
    const Array& h, /* Output at the previous time step */
    const Array& upwardWeight,
    const Array& upwardBias,
    const Array& literalWeight,
    const Array& literalBias) {

  Array cPrev = x.device().Dot(x, upwardWeight.Transpose()) + upwardBias +
    x.device().Dot(h, literalWeight.Transpose()) + literalBias;
  // States consists of a, i, f, o
  StackVector<Array> states = updateForward(extractGates(x));
  size_t batchSize = x.Shape()[0];
  Array cNext = states[0] * states[1] + states[2] * cPrev;
  Array h = states[3] * x.device().Tanh(cNext);
  return StackVector<Array>{cNext, h};
}
}

} // namespace native
} // namespace chainerx
