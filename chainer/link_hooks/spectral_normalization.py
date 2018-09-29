import numpy

from chainer import backend
from chainer.backends import cuda
from chainer import configuration
import chainer.functions as F
import chainer.links as L
from chainer import link_hook
from chainer import variable


def _l2normalize(v, eps=1e-12):
    xp = backend.get_array_module(v)
    if isinstance(xp, numpy):
        norm = numpy.sqrt(v * v + eps)
        return v / norm
    else:
        norm = cuda.reduce('T x', 'T out',
                           'x * x', 'a + b', 'out = sqrt(a)', 0,
                           'norm_sn')
        div = cuda.elementwise('T x, T norm, T eps',
                               'T out',
                               'out = x / (norm + eps)',
                               'div_sn')
        return div(v, norm(v), eps)


def calculate_max_singular_value(xp, W, u=None, n_power_iteration=1):
    """Calculate max singular value.

    Args:
        xp (numpy or cupy)
        W (chainer.variable.Parameter): Weight matrix.
        u (numpy.ndarray or cupy.ndarray): 1st singular vector.
        n_power_iteration (int): Number of power iteration.

    Returns:
        sigma (chainer.variable.Variable): max singular value.
        u (numpy.ndarray or cupy.ndarray): 1st singular vector.
        v (numpy.ndarray, cupy.ndarray): 2nd singular vector.

    """

    if u is None:
        u = xp.random.normal(size=(1, W.shape[0])).astype(xp.float32)
    u = u
    for _ in range(n_power_iteration):
        v = _l2normalize(xp.dot(u, W.array), eps=1e-12)
        u = _l2normalize(xp.dot(v, W.array.transpose()), eps=1e-12)
    sigma = F.sum(F.linear(u, F.transpose(W)) * v)
    return sigma, u


class SpectralNormalization(link_hook.LinkHook):
    """Spectral Normalization hook.

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

    .. admonition:: Example

        >>> layer = L.Convolution2D(3, 5, 3, 1, 1)
        >>> layer.add_hook(SpectralNormalization())
        >>> y = layer(np.random.uniform(-1, 1, (10, 3, 32, 32)).astype(np.float32))
        >>> layer = L.Deconvolution2D(6, 3, 4, 2, 1)
        >>> layer.add_hook(SpectralNormalization(1))
        >>> y = layer(np.random.uniform(-1, 1, (10, 6, 16, 16)).astype(np.float32))

    """

    name = 'SpectralNormalization'

    def __init__(self, axis=0, n_power_iteration=1, eps=1e-12, use_gamma=False,
                 factor=1.0, name=None):
        if n_power_iteration < 1:
            raise ValueError(
                'n_power_iteration is excepted to be positive.'
            )
        self.axis = axis
        self.n_power_iteration = n_power_iteration
        self.eps = eps
        self.use_gamma = use_gamma
        self.factor = factor
        self._initialied = False

        if name is not None:
            self.name = name

    def added(self, link):
        if isinstance(
            link,
            (L.Deconvolution1D, L.Deconvolution2D, L.Deconvolution3D, L.DeconvolutionND)):
            if self.axis != 1:
                raise ValueError('Invalid axis for Deconvolution layers.')
        if link.W.data is not None:
            self._prepare_parameters(link)
        u = link.xp.random.normal(size=(1, link.out_size)).astype(dtype='f')
        setattr(link, 'u', u)

    def deleted(self, link):
        del link.u
        if self.use_gamma:
            del link.gamma

    def forward_preprocess(self, cb_args):
        link = cb_args.link
        input_variable = cb_args.args[0]
        if not self._initialied:
            self._prepare_parameters(link, input_variable)

        self._normalize_weight(cb_args)

    def _normalize_weight(self, cb_args):
        link = cb_args.link
        W, u = link.W, link.u
        sigma, u = calculate_max_singular_value(link.xp, self._reshape_W(W), u)
        if configuration.config.train:
            link.u[:] = u

        W = self.factor * W / sigma
        if self.use_gamma:
            W *= F.broadcast_to(link.gamma[self.expander], W.shape)
        link.W = W

    def forward_postprocess(self, cb_args):
        pass

    def _prepare_parameters(self, link, input_variable=None):
        if link.W.array is None:
            if input_variable is None:
                return
            else:
                link._initialize_params(input_variable.shape[1])
        self._initialied = True
        expander = [None] * link.W.ndim
        expander[0] = Ellipsis
        self.expander = expander

        link.W_pre = link.W
        if self.use_gamma:
            weight_matrix = self._reshape_W(link.W.array)
            _, s, _ = link.xp.linalg.svd(weight_matrix)
            link.gamma = variable.Parameter(s[0])

    def _reshape_W(self, W):
        if self.axis != 0:
            axes = self.axis + [i for i in range(W.ndim) if i != self.axis]
            W = W.transpose(axes)
        return W.reshape(W.shape[0], -1)
