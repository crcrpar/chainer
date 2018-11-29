import numpy

import chainer
from chainer import backend
from chainer.backends import cuda
from chainer import configuration
import chainer.functions as F
import chainer.links as L
from chainer import link_hook
from chainer import variable


def l2normalize(xp, v, eps=1e-12):
    """Normalize a vector by L2 norm.

    Args:
        xp (numpy or cupy):
        v (numpy.ndarray or cupy.ndarray)
        eps (float): Epsilon value for numerical stability.

    Returns:
        normalized vector `v`

    """
    if xp is numpy:
        return v / (numpy.linalg.norm(v) + eps)
    else:
        norm = cuda.reduce('T x', 'T out',
                           'x * x', 'a + b', 'out = sqrt(a)', 0,
                           'norm_sn')
        div = cuda.elementwise('T x, T norm, T eps',
                               'T out',
                               'out = x / (norm + eps)',
                               'div_sn')
        return div(v, norm(v), eps)


def update_approximate_vectors(
        weight_matrix, u, n_power_iteration=1, eps=1e-12):
    """Update the first left singular vector `u`.

    This function updates using 2D weight and the first right singular
    vector `v`.  Note that this function is not backpropable.

    Args:
        weight_matrix (variable.Parameter): 2D weight.
        u (numpy.ndarray, cupy.ndarray, or None):
            Vector that has the shape of (1, out_size).
        n_power_iteration (int): Number of iterations to approximate
            the first right and left singular vectors.

    Returns:
        first left singular vector `u` and right singular vector `v`.

    """
    weight_matrix = weight_matrix.array  # No need to be backpropable.
    xp = backend.get_array_module(weight_matrix)
    for _ in range(n_power_iteration):
        v = l2normalize(xp, xp.dot(u, weight_matrix), eps)
        u = l2normalize(xp, xp.dot(v, weight_matrix.T), eps)
    return u, v


def calculate_max_singular_value(weight_matrix, u, v):
    """Calculate max singular value by power iteration method.

    Args:
        weight_matrix (chainer.Variable or chainer.Parameter)
        u (numpy.ndarray or cupy.ndarray)
        v (numpy.ndarray or cupy.ndarray)

    Returns:
        chainer.Variable of max singular value sigma

    """
    sigma = F.sum(F.linear(u, F.transpose(weight_matrix)) * v)
    return sigma


class SpectralNormalization(link_hook.LinkHook):
    """Spectral Normalization link hook implementation.

    SpectralNormalization is a weight normalization method which normalize
    weight matrix by maximum singular value. Max singular value is calculated
    by power iteration method and its 1st singular vector `u` is maintained
    by link.

    .. math::
         \mathbf{W} = \dfrac{\mathbf{W}}{\sigma(\mathbf{W})} \\
         \sigma(\mathbf{W}) = \max_{\mathbf{h}: \mathbf{h} \ne 0} \dfrac{\|\mathbf{W} \mathbf{h}\|_2}{\|\mathbf{h}\|_2}

    See: `Spectral Normalization for Generative Adversarial Networks\
          <https://arxiv.org/abs/1802.05957>`_

    Args:
        axis (int): Axis of weight which represents input variable size.
        n_power_iteration (int): Number of power iteration.
        eps (int): Numerical stability in norm calculation.
        use_gamma (bool): If ``True``, weight scaling parameter gamma which is
            initialized by initial weight's max singular value is introduced.
        factor (float): Scaling parameter of maximum singular value.
        name (str, None): Name of this hook. Default is 'SpectralNormalization'.
        dtype (numpy.dtype): Default is numpy.float32.

    .. admonition:: Example

        >>> layer = L.Convolution2D(3, 5, 3, 1, 1)
        >>> layer.add_hook(SpectralNormalization())
        >>> y = layer(np.random.uniform(-1, 1, (10, 3, 32, 32)).astype(np.float32))
        >>> layer = L.Deconvolution2D(6, 3, 4, 2, 1)
        >>> layer.add_hook(SpectralNormalization(axis=1))
        >>> y = layer(np.random.uniform(-1, 1, (10, 6, 16, 16)).astype(np.float32))
        >>> layer.delete_hook('SpectralNormalization')

    """

    name = 'SpectralNormalization'

    def __init__(self, axis=0, n_power_iteration=1, eps=1e-12, use_gamma=False,
                 factor=1.0, name=None, dtype=numpy.float32):
        assert n_power_iteration > 0
        self.axis = axis
        self.n_power_iteration = n_power_iteration
        self.eps = eps
        self.use_gamma = use_gamma
        self.factor = factor
        self.dtype = dtype
        self._initialied = False

        if name is not None:
            self.name = name

    def added(self, link):
        if isinstance(
            link, (
                L.Deconvolution1D, L.Deconvolution2D,
                L.Deconvolution3D, L.DeconvolutionND)):
            assert self.axis == 1
        if link.W.array is not None:
            # We cannot initialize buffers and parameters
            # before the link's weight is initialized.
            self._prepare_parameters(link)

    def deleted(self, link):
        # Re-register parameter W as a parameter.
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            new_weight = self.normalize_weight(link)
        current_weight = link.W
        delattr(link, 'W')
        param = chainer.Parameter(current_weight.array, new_weight.shape)
        with link.init_scope():
            link.W = param
        # Delete registered parameters and buffers.
        del link.W_pre
        del link.u
        if self.use_gamma:
            del link.gamma

    def forward_preprocess(self, cb_args):
        link = cb_args.link
        input_variable = cb_args.args[0]
        if not self._initialied:
            self._prepare_parameters(link, input_variable)

        normalizedW = self.normalize_weight(link)
        link.W = normalizedW

    def forward_postprocess(self, cb_args):
        # Nothing to do here.
        pass

    def _prepare_parameters(self, link, input_variable=None):
        # For clarity, we explicitly register unnormalized weight as W_pre.
        if link.W.array is None and input_variable is None:
            raise ValueError(
                "Either link's weight or input variable "
                "is assumed not to be None."
            )
        if link.W.array is None:
            # To copy the original weight, abuse this protected member method.
            link._initialize_params(input_variable.shape[1])
        assert link.W.array is not None, 'Failed to initialize W!'
        initialW = link.W
        del link.W
        # Add W as an attribute in order to skip `self._initialize_params`
        # executed by the given link's `__call__`.
        setattr(link, 'W', initialW)
        with link.init_scope():
            link.W_pre = initialW
        u = link.xp.random.normal(
            size=(1, initialW.shape[self.axis])).astype(dtype=self.dtype)
        link.u = u
        link.register_persistent('u')
        if self.use_gamma:
            # Initialize the scaling parameter `gamma` with a singular value.
            weight_matrix = self._reshape_W(initialW.array)
            _, s, _ = link.xp.linalg.svd(weight_matrix)
            gamma_shape = [1] * initialW.ndim
            with link.init_scope():
                link.gamma = variable.Parameter(s[0], gamma_shape)
        # Tell the initialization is Done.
        self._initialied = True

    def normalize_weight(self, link):
        # link = cb_args.link
        W_pre = link.W_pre
        u = link.u
        weight_matrix = self._reshape_W(W_pre)
        u, v = update_approximate_vectors(
            weight_matrix, u, self.n_power_iteration, self.eps)
        sigma = calculate_max_singular_value(weight_matrix, u, v) / self.factor
        if self.use_gamma:
            W = F.broadcast_to(link.gamma, W_pre.shape) * W_pre / sigma
        else:
            W = W_pre / sigma
        if configuration.config.train:
            link.xp.copyto(u, link.u)
        return W

    def _reshape_W(self, W):
        """Reshape weight into 2D if needed."""
        if W.ndim == 2:
            return W
        if self.axis != 0:
            axes = self.axis + [i for i in range(W.ndim) if i != self.axis]
            W = W.transpose(axes)
        return W.reshape(W.shape[0], -1)
