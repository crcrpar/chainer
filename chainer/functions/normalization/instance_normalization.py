from chainer import backend
from chainer.functions.array import reshape
from chainer.functions.array import tile
from chainer.functions.normalization import batch_normalization
from chainer.functions.normalization import group_normalization as gn_module
from chainer.utils import argument
from chainer.utils import type_check
from chainer import variable


def _check_type_and_shape(x, gamma, beta, mean=None, var=None):
    if x.ndim <= 2:
        raise ValueError('Input dimension must be greater than 2, '
                         'including batch size dimension '
                         '(first dimension).')
    n_channels = x.shape[1]
    xdtype = x.dtype
    if xdtype != gamma.dtype:
        raise type_check.InvalidType(expect=xdtype, actual=gamma.dtype)
    if xdtype != beta.dtype:
        raise type_check.InvalidType(expect=xdtype, actual=beta.dtype)
    if n_channels != gamma.size:
        raise type_check.InvalidType(expect=n_channels, actual=gamma.size)
    if n_channels != beta.size:
        raise type_check.InvalidType(expect=n_channels, actual=beta.size)
    if mean is not None:
        if xdtype != mean.dtype:
            raise type_check.InvalidType(expect=xdtype, actual=mean.dtype)
        if n_channels != mean.size:
            raise type_check.InvalidType(expect=n_channels, actual=mean.size)
    if var is not None:
        if xdtype != var.dtype:
            raise type_check.InvalidType(expect=xdtype, actual=var.dtype)
        if n_channels != var.size:
            raise type_check.InvalidType(expect=n_channels, actual=var.size)


def _tile(xp, x, reps, device):
    if isinstance(x, variable.Variable):
        x = tile.tile(x, reps)
    else:
        x = xp.tile(x, reps)
        x = device.send_array(x)
    return x


def _reshape_and_mean(xp, device, x, batch_size, channels):
    x = x.reshape(batch_size, channels)
    x = xp.array(x)
    return device.send_array(x.mean(axis=0))


def instance_normalization(x, gamma, beta, **kwargs):
    """Instance normalization function.

    This function implements instance normalization
    which normalizes each sample by its mean and standard deviation.

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`): Batch tensors.
            First dimension of this value must be the size of minibatch and
            second dimension must be the number of channels.
            Moreover, this value must have one or more following dimensions,
            such as height and width.
        gamma (:class:`~chainer.Variable` or :ref:`ndarray`):
            Scaling parameter of normalized data with the shape of (C,).
        beta (:class:`~chainer.Variable` or :ref:`ndarray`):
            Shifting parameter of normalized data with the shape o f(C,).
        eps (float): Epsilon value for numerical stability.
        running_mean (:class:`~chainer.Variable`, :ref:`ndarray`, or None):
            Shifting parameter of input with the shape of (C,).
        running_var (:class:`~chainer.Variable`, :ref:`ndarray`, or None):
            Scaling parameter of input with the shape of (C,).
        decay (float): Decay rate of moving average. It is used during
            training.

    Returns:
        :class:`~chainer.Variable`: The output variable which has the same
        shape as :math:`x`.

    See: `Instance Normalization: The Missing Ingredient for Fast Stylization
           <https://arxiv.org/abs/1607.08022>`_

    """
    if x.ndim < 3:
        raise ValueError(
            'Instance Normalization requires ``x`` to be at least 3-d.'
        )
    eps, running_mean, running_var, decay = argument.parse_kwargs(
        kwargs, ('eps', 2e-5), ('running_mean', None),
        ('running_var', None), ('decay', 0.9)
    )
    _check_type_and_shape(x, gamma, beta, running_mean, running_var)
    channels = x.shape[1]
    gn = gn_module.GroupNormalization(
        channels, eps, running_mean, running_var, decay)
    y, = gn.apply((x, gamma, beta))
    return y


def fixed_instance_normalization(x, gamma, beta, mean, var, eps=2e-5):
    """Instance Normalization with fixed statistics.

    This is a variant of instance normalization, where the mean and variance
    are given by the caller as fixed variables. This is used on testing mode
    of the instance normalization layer with
    ``track_running_stats`` of ``True``.

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.
        gamma (:class:`~chainer.Variable` or :ref:`ndarray`):
            Scaling parameter of normalized data with the shape of (C,).
        beta (:class:`~chainer.Variable` or :ref:`ndarray`):
            Shifting parameter of normalized data with the shape o f(C,).
        mean (:class:`~chainer.Variable` or :ref:`ndarray`):
            Shifting parameter of input with the shape of (C,).
        var (:class:`~chainer.Variable` or :ref:`ndarray`):
            Scaling parameter of input with the shape of (C,).
        eps (float): Epsilon value for numeircal stability.

    .. seealso::
       :func:`~chainer.functions.instance_normalization`,
       :class:`~chainer.links.InstancNormalization`

    """
    _check_type_and_shape(x, gamma, beta, mean, var)
    original_shape = x.shape
    batch_size, channels = original_shape[:2]
    x = reshape.reshape(x, (1, batch_size * channels) + original_shape[2:])

    gamma = tile.tile(gamma, batch_size)
    beta = tile.tile(beta, batch_size)
    tiled_mean = tile.tile(mean, batch_size)
    tiled_var = tile.tile(var, batch_size)

    y = batch_normalization.fixed_batch_normalization(
        x, gamma, beta, tiled_mean, tiled_var, eps=eps,
    )
    y = reshape.reshape(y, original_shape)
    return y
