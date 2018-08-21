import warnings

import numpy

import chainer
from chainer.backends import cuda
from chainer.backends import intel64
from chainer import configuration
from chainer import function
from chainer import function_node
from chainer import functions
from chainer.functions.normalization import batch_normalization
from chainer.utils import argument
from chainer.utils import collections_abc
from chainer.utils import type_check


if cuda.cudnn_enabled:
    cudnn = cuda.cudnn
    libcudnn = cuda.cuda.cudnn
    _cudnn_version = cuda.cuda.cudnn.getVersion()


def _compute_axis(x_ndim, gamma_ndim=1, axis=None):
    if axis is None:
        axis = (0,) + tuple(range(gamma_ndim + 1, x_ndim))
    return axis


# Computes a complementary set of axis
def _compute_key_axis(x_ndim, gamma_ndim=1, axis=None):
    axis = _compute_axis(x_ndim, gamma_ndim, axis)
    key_axis = tuple([i for i in range(x_ndim) if i not in axis])
    return key_axis


class InstanceNormalization(functions.BatchNormalization):

    mean = None
    inv_std = None
    axis = None

    def __init__(self, eps=2e-5, mean=None, var=None, decay=0.9):
        self.running_mean = mean
        self.running_var = var

        axis = InstanceNormalization.axis
        # Note: cuDNN requires that eps be greater than or equals to
        # CUDNN_BN_MIN_EPSILON. Otherwise, an error will occur.
        # See CUDNN_BN_MIN_EPSILON value in cudnn.h to verify minimum allowable
        # value.
        self.eps = eps
        if chainer.should_use_cudnn('>=auto'):
            if eps < libcudnn.CUDNN_BN_MIN_EPSILON:
                raise RuntimeError(
                    'cuDNN does not allow an eps value '
                    'less than {}.'.format(libcudnn.CUDNN_BN_MIN_EPSILON))
        self.decay = decay
        if isinstance(axis, collections_abc.Sequence):
            for i in range(1, len(axis)):
                if axis[i - 1] >= axis[i]:
                    msg = 'numbers in axis must be sorted in ascending order'
                    raise RuntimeError(msg)
        elif isinstance(axis, int):
            axis = axis,
        elif axis is not None:
            raise RuntimeError('axis must be int, tuple of int or None')
        self.axis = axis

    def forward(self, inputs):
        self.retain_inputs((0, 1, 3, 4))
        x, gamma, beta, mean, var = inputs

        xp = cuda.get_array_module(x)
        if self.running_mean is None:
            self.running_mean = xp.zeros_like(gamma)
            self.running_var = xp.zeros_like(gamma)

        self.axis = _compute_axis(x.ndim, gamma.ndim, self.axis)
        self.key_axis = _compute_key_axis(x.ndim, gamma.ndim, self.axis)

        org_shape = x.shape
        batchsize, channels, *res = org_shape
        x = xp.reshape(x, [1, batchsize * channels] + res)
        gamma = xp.repeat(gamma, batchsize, 0)
        beta = xp.repeat(beta, batchsize, 0)
        running_mean = xp.repeat(self.running_mean, batchsize, 0)
        running_var = xp.repeat(self.running_var, batchsize, 0)

        expander = [None for _ in range(x.ndim)]
        for i in self.key_axis:
            expander[i] = slice(None)
        expander = tuple(expander)
        self.expander = expander

        self.mode = _INMode(x, gamma, self.key_axis)
        self.use_cudnn = self.mode.can_use_cudnn(xp)
        self.use_ideep = self.mode.can_use_ideep()

        if self.use_ideep:
            expand_dim = False
            if x.ndim == 2:
                expand_dim = True
                x = x[:, :, None, None]

            gamma = gamma[expander]
            beta = beta[expander]
            W = numpy.concatenate((gamma, beta), axis=0).reshape((2, -1))

            y, mean, var, inv_std = (
                intel64.ideep.batchNormalization.Forward(
                    intel64.ideep.array(x),
                    intel64.ideep.array(W),
                    None,
                    None,
                    self.eps
                ))
            y = xp.reshape(y, org_shape)
            self.mean = xp.mean(xp.reshape(mean, (batchsize, channels)), 0)
            self.var = xp.mean(xp.reshape(var, (batchsize, channels)), 0)
            self.inv_std = xp.mean(xp.reshape(inv_std, (batchsize, channels)), 0)

            m = x.size // gamma.size
            adjust = m / max(m - 1., 1.)

            # Update running_mean
            if isinstance(self.running_mean, intel64.ideep.mdarray):
                self.running_mean.inplace_axpby(
                    self.decay, (1 - self.decay), self.mean)
            else:
                self.running_mean *= self.decay
                self.running_mean += self.mean * (1 - self.decay)

            # Update running_var
            if isinstance(self.running_var, intel64.ideep.mdarray):
                self.running_var.inplace_axpby(
                    self.decay, (1 - self.decay), self.var * adjust)
            else:
                self.running_var *= self.decay
                self.running_var += self.var * adjust * (1 - self.decay)

            if expand_dim:
                y = numpy.squeeze(y, axis=(2, 3))

        elif self.use_cudnn:
            x = cuda.cupy.ascontiguousarray(x)

            gamma = cuda.cupy.ascontiguousarray(gamma)
            beta = cuda.cupy.ascontiguousarray(beta)
            dtype = x.dtype
            handle = cudnn.get_handle()
            x_desc = cudnn.create_tensor_descriptor(
                _as4darray(x, self.mode))
            cudnn_mode = self.mode.get_cudnn_mode()
            derivedBnDesc = cudnn.create_uninitialized_tensor_descriptor()
            libcudnn.deriveBNTensorDescriptor(derivedBnDesc.value,
                                              x_desc.value, cudnn_mode)
            dtype_param = _get_dtype_of_tensor_descriptor(derivedBnDesc)
            running_mean = xp.repeat(self.running_mean, batchsize, 0)
            running_var = xp.repeat(self.running_var, batchsize, 0)
            if dtype_param is not dtype:
                gamma = gamma.astype(dtype_param)
                beta = beta.astype(dtype_param)
                running_mean = running_mean.astype(dtype_param)
                running_var = running_var.astype(dtype_param)
            else:
                running_mean = running_mean
                running_var = running_var

            oz_dtype = 'd' if x.dtype == 'd' else 'f'
            one = numpy.array(1, dtype=oz_dtype).ctypes
            zero = numpy.array(0, dtype=oz_dtype).ctypes
            y = cuda.cupy.empty_like(x)
            # Factor used in the moving average
            factor = 1 - self.decay

            if self.mean is None:
                self.mean = xp.empty_like(gamma)
                self.inv_std = xp.empty_like(gamma)

            mean = xp.repeat(self.mean, batchsize, 0)
            inv_std = xp.repeat(self.inv_std, batchsize, 0)
            libcudnn.batchNormalizationForwardTraining(
                handle, cudnn_mode, one.data, zero.data,
                x_desc.value, x.data.ptr, x_desc.value,
                y.data.ptr, derivedBnDesc.value, gamma.data.ptr,
                beta.data.ptr, factor, running_mean.data.ptr,
                running_var.data.ptr, self.eps,
                mean.data.ptr, inv_std.data.ptr)

            if (cudnn_mode is libcudnn.CUDNN_BATCHNORM_SPATIAL_PERSISTENT and
                    configuration.config.debug):
                query_mode = libcudnn.CUDNN_ERRQUERY_BLOCKING
                rstatus = libcudnn.queryRuntimeError(handle, query_mode)
                if rstatus is not libcudnn.CUDNN_STATUS_SUCCESS:
                    warnings.warn(
                        'A numerical overflow might have happend in cuDNN'
                        'batch normalization (status:{})'.format(rstatus))

            running_mean = xp.mean(xp.reshape(running_mean, (batchsize, channels)), 0)
            running_var = xp.mean(xp.reshape(running_var, (batchsize, channels)), 0)
            if dtype_param is not dtype:
                running_mean = running_mean.astype(dtype)
                running_var = running_var.astype(dtype)
                self.running_mean.data.copy_from(running_mean.data,
                                                 running_mean.nbytes)
                self.running_var.data.copy_from(running_var.data,
                                                running_var.nbytes)
        else:
            # Generic CPU and GPU implementation
            # FIXME (kozuki)

            gamma = gamma[expander]
            beta = beta[expander]
            self.mean = x.mean(axis=self.axis)
            var = x.var(axis=self.axis)
            if xp is numpy:
                self.inv_std = numpy.reciprocal(numpy.sqrt(
                    var + self.eps, dtype=x.dtype))
            else:
                self.inv_std = cuda.cupyx.rsqrt(var + self.eps)
            y = _apply_in_fwd(xp, x, self.mean[expander],
                              self.inv_std[expander], gamma, beta)
            # Update running statistics
            m = x.size // gamma.size
            adjust = m / max(m - 1., 1.)  # unbiased estimation
            self.running_mean *= self.decay
            self.running_mean += (1 - self.decay) * self.mean
            self.running_var *= self.decay
            self.running_var += (1 - self.decay) * adjust * var

        return y,

    def backward(self, indexes, grad_outputs):
        x, gamma = self.get_retained_inputs()
        gy, = grad_outputs

        if self.use_ideep:
            assert self.var is not None
            var = self.var
        else:
            var = None

        f = batch_normalization.BatchNormalizationGrad(
            self.eps, self.use_cudnn, self.mode, self.expander, self.axis,
            self.mean, var, self.inv_std, self.key_axis)
        return f(x, gamma, gy)


def _apply_in_fwd(xp, x, mean, inv_std, gamma, beta):
    # NOTE: all arguments should be broadcasted to x.shape
    # (mean, inv_std, gamma, and beta have to already be expanded)
    if xp is numpy:
        x_hat = _x_hat(x, mean, inv_std)
        y = gamma * x_hat
        y += beta
    else:
        y = cuda.elementwise(
            'T x, T mean, T inv_std, T gamma, T beta', 'T y',
            'y = gamma * (x - mean) * inv_std + beta', 'bn_fwd'
        )(x, mean, inv_std, gamma, beta)
    return y


class _INMode(object):
    """Derived from chainer/functions/normalization/batch_normalization.py."""

    def __init__(self, x, gamma, key_axis, inference=False):
        is_gamma_1d = gamma.ndim == 1
        # cuDNN only supports these tensor dimensions because they are
        # the most commonly used. If there is a need to support other
        # dimensions with cuDNN, we could consider reshaping the input
        # into a 2-dim array with channels as second dim and m=<product
        # of all dimensions except the 2nd dimension> as the first
        # dimension.
        self.is_for_conv2d = is_gamma_1d and x.ndim == 4 and key_axis[0] == 1
        self.is_for_linear = is_gamma_1d and key_axis[0] == x.ndim - 1
        self.cudnn_dim_ok = self.is_for_conv2d or self.is_for_linear
        # self.cudnn_dtype_ok = x.dtype != numpy.float16
        self.cudnn_dtype_ok = self.is_for_conv2d or (x.dtype != numpy.float16)
        self.ideep_ok = is_gamma_1d and intel64.inputs_all_ready((x,))
        self.inference = inference

    def get_cudnn_mode(self):
        assert self.cudnn_dim_ok
        if self.is_for_linear:
            return libcudnn.CUDNN_BATCHNORM_PER_ACTIVATION

        if (not self.inference and _cudnn_version >= 7000 and
                configuration.config.cudnn_fast_batch_normalization):
            return libcudnn.CUDNN_BATCHNORM_SPATIAL_PERSISTENT
        return libcudnn.CUDNN_BATCHNORM_SPATIAL

    def can_use_ideep(self):
        return self.ideep_ok and intel64.should_use_ideep('>=auto')

    def can_use_cudnn(self, xp):
        # TODO(bkvogel): Check for float16 support again in next cuDNN version.
        # cuDNN v5 batch normalization does not seem to support float16.
        return (xp is not numpy and
                chainer.should_use_cudnn('>=auto', 5000) and
                self.cudnn_dim_ok and
                self.cudnn_dtype_ok)


def _as4darray(arr, mode):
    assert mode.cudnn_dim_ok
    if mode.is_for_conv2d:
        assert arr.ndim == 4
        return arr
    else:  # is_for_linear
        return arr.reshape(numpy.prod(arr.shape[0:-1]), -1, 1, 1)


def _x_hat(x, mean, inv_std):
    x_mu = x - mean
    x_mu *= inv_std
    return x_mu


def _apply_in_fwd(xp, x, mean, inv_std, gamma, beta):
    # NOTE: all arguments should be broadcasted to x.shape
    # (mean, inv_std, gamma, and beta have to already be expanded)
    if xp is numpy:
        x_hat = _x_hat(x, mean, inv_std)
        y = gamma * x_hat
        y += beta
    else:
        y = cuda.elementwise(
            'T x, T mean, T inv_std, T gamma, T beta', 'T y',
            'y = gamma * (x - mean) * inv_std + beta', 'bn_fwd'
        )(x, mean, inv_std, gamma, beta)
    return y


def _zero_if_none(xp, x, shape, dtype):
    # TODO(Tokui): Return broadcasted 0 instead of a zeroed array.
    if x is None:
        return xp.zeros(shape, dtype=dtype)
    return x


def _get_dtype_of_tensor_descriptor(desc):
    cudnn_dtype, _, _, _, _, _, _, _, _ = libcudnn.getTensor4dDescriptor(
        desc.value)
    dtype = None
    if cudnn_dtype == libcudnn.CUDNN_DATA_DOUBLE:
        dtype = numpy.dtype(numpy.float64)
    elif cudnn_dtype == libcudnn.CUDNN_DATA_FLOAT:
        dtype = numpy.dtype(numpy.float32)
    elif cudnn_dtype == libcudnn.CUDNN_DATA_HALF:
        dtype = numpy.dtype(numpy.float16)
    else:
        msg = 'Unknow cudnn data type {} '.format(cudnn_dtype)
        raise RuntimeError(msg)
    return dtype


def instance_normalization(x, gamma, beta, **kwargs):
    """instance_normalization(x, gamma, beta, eps=2e-5, running_mean=None, running_var=None, decay=0.9)

    Instance normalization function.

    It takes the input variable ``x`` and two parameter variables ``gamma`` and
    ``beta``. The parameter variables must both have the same dimensionality,
    which is referred to as the channel shape. This channel shape corresponds
    to the dimensions in the input which are not averaged over. Since the
    first dimension of the input corresponds to the batch size, the second
    dimension of ``x`` will correspond to the first dimension of the channel
    shape, the third dimension of ``x`` will correspond to the second channel
    dimension (if it exists) and so on. Therefore, the dimensionality of the
    input must be at least one plus the number of channel dimensions. The
    total effective "batch size" will then be considered to be the product of
    all dimensions in ``x`` except for the channel dimensions.

    As an example, if the input is four dimensional and the parameter
    variables are one dimensional, then it is assumed that the first
    dimension of the input is the batch size, the second dimension is the
    channel size, and the remaining two dimensions are considered
    to be spatial dimensions that will be averaged over along with the
    batch size in the batch normalization computations. That is,
    the total batch size will be considered to be the product of all
    input dimensions except the second dimension.

    .. warning::

       ``train`` argument is not supported anymore since v2.
       Instead, use ``chainer.using_config('train', train)``.
       See :func:`chainer.using_config`.

    Args:
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Input variable.
        gamma (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Scaling parameter of normalized data.
        beta (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Shifting parameter of scaled normalized data.
        eps (float): Epsilon value for numerical stability.
        running_mean (numpy.ndarray or cupy.ndarray):
            Running average of the mean. This is a running average of
            the mean over several mini-batches using the decay parameter.
            The function takes a previous running average, and updates
            the array in-place by the new running average.
            If ``None``, the running average is not computed. If this is
            ``None``, then ``runnng_var`` must also be ``None``.
        running_var (numpy.ndarray or cupy.ndarray):
            Running average of the variance. This is a running average of
            the variance over several mini-batches using the decay parameter.
            The function takes a previous running average, and updates
            the array in-place by the new running average.
            If ``None``, the running average is not computed. If this is
            ``None``, then ``running_mean`` must also be ``None``.
        decay (float): Decay rate of moving average. It is used during
            training.

    See: `Instance Normalization: The Missing Ingredient for Fast Stylization\
          <https://arxiv.org/abs/1607.08022>`_

    .. seealso:: :class:`~chainer.links.BatchNormalization`

    """  # NOQA

    eps, running_mean, running_var, decay = argument.parse_kwargs(
        kwargs, ('eps', 2e-5), ('running_mean', None),
        ('running_var', None), ('decay', 0.9),
        train='train argument is not supported anymore. '
        'Use chainer.using_config')

    return InstanceNormalization(eps, running_mean, running_var, decay,
                              ).apply((x, gamma, beta))[0]
