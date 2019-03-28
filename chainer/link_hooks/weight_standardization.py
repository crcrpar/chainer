import numpy

import chainer
from chainer import backend
from chainer import function_node
from chainer import link_hook


class WeightStandardization(link_hook.LinkHook):

    name = 'WeightStandardization'

    def __init__(self, axis=0, eps=1e-5, weight_name='W', name=None):
        if axis != 0:
            raise ValueError('Invalid axis value.')
        self.axis = axis
        self.eps = eps
        self.weight_name = weight_name

    def added(self, link):
        if not isinstance(link, chainer.links.Convolution2D):
            raise NotImplementedError('Only supports L.Convolution2D')
        self._initialized = getattr(link, self.weight_name).array is not None

    def __enter__(Self):
        raise NotImplementedError(
            'This hook is not supposed to be used as context manager.')

    def forward_preprocess(self, cb_args):
        link = cb_args.link
        input_variable = cb_args.args[0]
        if not self._initialized:
            if getattr(link, self.weight_name).array is None:
                if input_variable is not None:
                    link._initialize_params(input_variable.shape[1])
        weight = getattr(link, self.weight_name)
        self.original_weight = weight
        # note: `normalized_weight` is ~chainer.Variable
        normalized_weight = self.normalize_weight(weight)
        setattr(link, self.weight_name, normalized_weight)

    def forward_postprocess(self, cb_args):
        # Here, the computational graph is already created,
        # we can reset link.W or equivalents to be Parameter.
        link = cb_args.link
        setattr(link, self.weight_name, self.original_weight)

    def normalized_weight(self, weight):
        return standardize_weight(weight, axis=0, eps=self.eps)


class StandardizeWeight(function_node.FunctionNode):

    def __init__(self, axis=0, eps=1e-5):
        self.axis = axis
        self.eps = eps

    def forward(self, inputs):
        self.retain_inputs((0,))
        W = inputs[0]
        xp = backend.get_array_module(W)
        self.orig_shape = W.shape
        out_size = self.orig_shape[self.axis]
        self.shape_2d = (out_size, numpy.prod(self.orig_shape) // out_size)

        expander = (Ellipsis, None)
        W = self.reshape_W(xp, W)
        self.mean = xp.mean(W, axis=1)
        W -= self.mean[expander]
        self.std = xp.std(W, axis=1)
        W /= (self.std + self.eps)[expander]
        W = self.re_reshape_W(self, xp, W)
        return W,

    def reshape_W(self, xp, W):
        if self.axis == 0:
            return xp.reshape(W, self.shape_2d)
        self.axes = [self.axis] + [
            i for i in range(len(self.orig_shape)) if i != self.axis]
        W = xp.transpose(W, self.axes)
        return xp.reshape(W, self.shape_2d)

    def re_reshape_W(self, xp, W):
        if self.axis == 0:
            return xp.reshape(W, self.orig_shape)
        tmp_shape = [self.orig_shape[i] for i in self.axes]
        W = xp.transpose(xp.reshape(W, tmp_shape), self.axes)
        return W

    def backward(self, indexes, grad_outputs):
        W, = self.get_retained_inputs()
        gy, = grad_outputs
        xp = backend.get_array_module(gy)
        reshaped_gy = self.reshape_W(xp, gy)
        reshaped_gy /= (self.std + self.eps)[Ellipsis, None]
        return self.re_reshape_W(xp, reshaped_gy)


def standardize_weight(W, axis, eps):
    return StandardizeWeight(eps, axis).apply((W,))[0]
