import numpy

from chainer import backend
from chainer.functions.array import expand_dims
from chainer.functions.math import minimum
from chainer.functions.math import sign
from chainer import function_node
from chainer import utils
from chainer.utils import type_check


def _forward_generic(x, t, norm, reduce):
    xp = backend.get_array_module(x)
    mask = xp.zeros_like(x)
    mask[:, t] = -1
    tmp = xp.copy(x)
    tmp[:, t] = numpy.finfo(x.dtype).min
    mask[:, xp.argmax(tmp, 1)] = 1
    margin = xp.maximum(0, 1 + xp.sum(mask * x, 1))

    if norm == "L1":
        loss = margin
    elif norm == "L2":
        loss = margin ** 2
    else:
        # norm == Huber
        quad = (margin < 2).astype(x.dtype)
        loss = margin ** 2 / 4 * quad + (margin - 1) * (1 - quad)

    if reduce == "mean":
        loss = utils.force_array(xp.sum(loss) / len(x), dtype=x.dtype)
    return loss, mask, margin


class HingeMaxMargin(function_node.FunctionNode):

    """Hinge max margin loss."""

    def __init__(self, norm="L2", reduce="mean"):
        if norm in ["L1", "L2", "Huber"]:
            self.norm = norm
        else:
            raise NotImplementedError(
                "norm should be either 'L1', 'L2' or 'Huber'")

        if reduce in ["mean", "along_second_axis"]:
            self.reduce = reduce
        else:
            raise ValueError(
                "only 'mean' and 'along_second_axis' are valid for 'reduce',"
                " but '%s' is "
                "given" % reduce
            )
        self.mask = None
        self.margin = None

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)

        x_type, t_type = in_types
        type_check.expect(
            x_type.dtype.kind == "f",
            t_type.dtype.kind == "i",
            x_type.ndim == 2,
            t_type.ndim == 1,
            x_type.shape[0] == t_type.shape[0],
        )

    def forward_chainerx(self, inputs):
        x, t = inputs
        loss, self.mask, self.margin = _forward_generic(
            x, t, self.norm, self.reduce)
        return (loss,)

    def forward(self, inputs):
        x, t = inputs
        loss, self.mask, self.margin = _forward_generic(
            x, t, self.norm, self.reduce)

        return (loss,)

    def backward(self, indexes, grad_outputs):
        gloss, = grad_outputs

        if self.reduce == "mean":
            gloss /= self.margin.shape[0]

        if self.norm == "L1":
            gx = (
                gloss * sign.sign(
                    self.mask * expand_dims.expand_dims(self.margin, 1)))
        elif self.norm == "L2":
            gx = (
                2 * gloss * self.mask
                * expand_dims.expand_dims(self.margin, 1))
        elif self.norm == "Huber":
            gx = (
                gloss
                * self.mask
                * expand_dims.expand_dims(
                    minimum.minimum(self.margin / 2, sign.sign(self.margin)), 1
                )
            )
        else:
            raise NotImplementedError()

        return gx, None


def hinge_max_margin(x, t, norm="L2", reduce="mean"):
    """Computes the hinge loss for a one vs max classification task.

    .. math::
        margin_{i} = ReLU \\left(1 - x_{i,t_{i}} + max_{k, k \\neq t_{i}}
         \\left(x_{i, k} \\right) \\right)

    and

    .. math::
        loss_{i} = \\left \\{
         \\begin{array}{cc}
         margin_{i}     & {\\rm if~norm} = {\\rm L1} \\\\
         margin_{i}^{2} & {\\rm if~norm} = {\\rm L2} \\\\
         margin_{i}-1   & {\\rm if~norm} =
         {\\rm Huber \\& margin_{i} \\ge 2} \\\\
         margin_{i}^{2} & {\\rm if~norm} ={\\rm Huber \\& margin_{i} < 2}
         \\end{array} \\right \\}

    All 3 norms are continuous. ``'L2'`` and ``'Huber'`` are differentiable,
    ``'L1'`` is not.

    The output is a variable whose value depends on the value of
    the option ``reduce``. If it is ``'no'``,
    it holds the loss values for each example. If it is ``'mean'``,
    it takes the mean of loss values.

    See:
        - `Huber loss - Wikipedia
          <https://en.wikipedia.org/wiki/Huber_loss>`_
        - `Structured support vector machine - Wikipedia \
<https://en.wikipedia.org/wiki/Structured_support_vector_machine>`_

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.
            The shape of ``x`` should be (:math:`N`, :math:`K`).
        t (:class:`~chainer.Variable` or :ref:`ndarray`): The
            :math:`N`-dimensional label vector with values
            :math:`t_n \\in \\{0, 1, 2, \\dots, K-1\\}`.
            The shape of ``t`` should be (:math:`N`,).
        norm (str): Specifies norm type. Either ``'L1'`` , ``'L2'`` ,
            ``'Huber'`` are acceptable.
        reduce (str): Reduction option. Its value must be either
            ``'mean'`` or ``'no'``. The default value is ``'mean'``.

    Returns:
        ~chainer.Variable:
            A variable object holding the hinge max margin loss.
            If ``reduce`` is ``'no'``, the output variable holds
            an array whose shape is same :math:`N`.
            If it is ``'mean'``, the output variable holds a scalar value.

    .. admonition:: Example

        >>> # the batchsize is 4 and the number of classes is 2.
        >>> import numpy as np
        >>> import chainer.functions as F
        >>> x = np.stack(
        ...     (np.arange(10), 5 * np.ones(10)), 1).astype(np.float32)
        >>> t = np.ones((10,), np.int32)
        >>> F.hinge_max_margin(x, t, norm='L1', reduce='no')
        variable([0., 0., 0., 0., 0., 1., 2., 3., 4., 5.])
        >>> F.hinge_max_margin(x, t)
        variable(5.5)
        >>> F.hinge_max_margin(x, t, norm='L1')
        variable(1.5)
        >>> F.hinge_max_margin(x, t, norm='Huber')
        variable(1.025)

    """
    return HingeMaxMargin(norm, reduce).apply((x, t))[0]
