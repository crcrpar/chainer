import chainer
from chainer import link_hook
from chainer import functions as F
from chainer import links as L
from chainer import variable


def _get_expander(ndim, axis):
    expander = [None] * ndim
    expander[axis] = Ellipsis
    return expander


def _norm_axis(weight, reduce_axes, eps=1e-12, keepdims=True):
    # reduce_axes = tuple([i for i in range(weight.ndim) if i != axis])
    reduce_axes = tuple(reduce_axes)
    squared_norm = F.sum(weight * weight, axis=reduce_axes, keepdims=keepdims)
    norm = F.sqrt(squared_norm + eps)
    return norm


def _expand_and_broadcast(var, expand_axes, shape):
    for i in expand_axes:
        var = F.expand_dims(var, i)
    return F.broadcast_to(var, shape)


class WeightNormalization(link_hook.LinkHook):
    """Weight Normalization hook.

    Weight normalization is a reparameterization of the weight vectors
    of a link decoupling the length of those weight vectors
    from their direction.
    This hook registers the weight parameter as `_V` and
    scaling parameter `g` to the link when the :meth:`added` is called.
    When this hook is removed from the link, this hook removes `g` and `_V`
    from that.

    .. math:: \mathbf{W} = g \dfrac{\mathbf{V}}{\|\mathbf{v}\|}

    See: `Weight Normalization: A Simple Reparameterization to Accelerate\
          Training of Deep Neural Networks <https://arxiv.org/abs/1602.07868>`_

    Args:
        axis (int): Axis of weight which represents input variable size.
        eps (int): Numerical stability in norm calculation.

    .. admonition:: Example

        >>> layer = L.Convolution2D(3, 5, 3, 1, 1)
        >>> layer.add_hook(WeightNormalization())
        >>> y = layer(np.random.uniform(-1, 1, (10, 3, 32, 32)).astype('f'))
        >>> layer = L.Deconvolution2D(6, 3, 4, 2, 1)
        >>> layer.add_hook(WeightNormalization(1))
        >>> y = layer(np.random.uniform(-1, 1, (10, 6, 16, 16)).astype('f'))

    """

    name = 'WeightNormalization'

    def __init__(self, axis=0, eps=1e-12, name=None):
        self.axis = axis
        self.eps = eps
        self._initialied = False

        if name is not None:
            self.name = name

    def added(self, link):
        if isinstance(
            link,
            (L.Deconvolution1D, L.Deconvolution2D, L.Deconvolution3D, L.DeconvolutionND)):
            if self.axis == 0:
                raise ValueError("Wrong axis for Deconvolution layer.")
        self._prepare_parameters(link)

    def deleted(self, link):
        del link.g
        del link._V

    def forward_preprocess(self, cb_args):
        link = cb_args.link
        if not self._initialied:
            input_variable = cb_args.args[0]
            self._prepare_parameters(link, input_variable)
            print('Initialized weights.')

        self._reparameterize_weight(link)

    def forward_postprocess(self, cb_args):
        pass

    def _prepare_parameters(self, link, input_variable=None):
        if link.W.array is None:
            if input_variable is None:
                return
            else:
                link._initialize_params(input_variable.shape[1])
        W = link.W
        expander = _get_expander(W.ndim, self.axis)
        self.expand_axes = tuple([i for i, d in enumerate(expander) if d is None])
        with chainer.no_backprop_mode():
            g_variable = _norm_axis(W, self.expand_axes, self.eps)
        g = variable.Parameter(g_variable.array)
        with link.init_scope():
            link.g = g
            link._V = W
        self.shape = W.shape
        print(self.shape, self.expand_axes)
        self._initialied = True

    def _reparameterize_weight(self, link):
        weight = link._V
        norm = _norm_axis(link._V, self.expand_axes, self.eps)
        norm = F.broadcast_to(norm, self.shape)
        g = F.broadcast_to(link.g, self.shape)
        link.W = g * weight / norm
