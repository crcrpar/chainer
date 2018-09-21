import chainer
from chainer.backends import cuda
from chainer import link_hook
from chainer import links as L
from chainer import variable


class WeightNormalization(link_hook.LinkHook):
    """Weight Normalization hook.

    Paper:
    Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks
    https://arxiv.org/abs/1602.07868

    Args:
        axis (int): Input side axis. Default is 0 but note that deconvolution set input axis 1.
        eps (int): Numerical stability in norm calculation.

    Attributes:
        g (chainer.variable.Parameter)

    """

    name = 'WeightNormalization'

    def __init__(self, axis=0, eps=1e-12):
        self.axis = axis
        self.eps = eps
        self._initialied = False
        self._is_deconv = False
        self.ndim = -1

    def added(self, link):
        if isinstance(link, (L.Deconvolution2D, L.DeconvolutionND)):
            self._is_deconv = True
            if self.axis == 0:
                raise ValueError(
                    "Wrong axis for {} instance".format(type(link))
                )
        self._prepare_parameters(link)

    def deleted(self, link):
        del link.g
        del link._V

    def forward_preprocess(self, cb_args):
        link = cb_args.link
        if not self._initialied:
            input_variable = cb_args.args[0]
            self._prepare_parameters(link, input_variable)

        self._reparameterize_weight(link)

    def forward_postprocess(self, cb_args):
        pass

    def _prepare_parameters(self, link, input_variable=None):
        if link.W is None:
            if input_variable is None:
                return
            else:
                link._initialize_params(input_variable.shape[1])
        self._initialied = True
        W = link.W
        self.ndim = W.ndim
        g = link.xp.random.randn(W.shape[self.axis]).astype(link.xp.float32)
        g /= link.xp.linalg.norm(axis=self.axis)
        link.g = variable.Parameter(g)
        link._V = W

    def _reparameterize_weight(self, link, input_variable):
        V = link._V
        g = link.g
        xp = link.xp
        ndim = self.ndim

        aggr_axes = []
        expand_axes = [None] * ndim
        for i in range(ndim):
            if i != self.axis:
                aggr_axes.append(i)
            else:
                expand_axes[i] = Ellipsis

        norm = xp.sqrt(xp.sum(V * V, axis=aggr_axes, keepdims=True) + self.eps)
        link.W[...] = g * V / norm
