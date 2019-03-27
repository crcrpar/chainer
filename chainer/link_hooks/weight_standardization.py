import chainer
from chainer import functions
from chainer import link_hook


class WeightStandardization(link_hook.LinkHook):

    name = 'WeightStandardization'

    def __init__(self, eps=1e-5, weight_name='W', name=None):
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
        # This method normalizes target link's weight spectrally
        # using power iteration method
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
        expander = (Ellipsis, None, None, None)
        orig_shape = weight.shape
        reshaped = functions.reshape(weight, (orig_shape[0], -1))
        mean = functions.mean(reshaped, 1)
        reshaped = reshaped - mean[expander]
        reshaped_std = functions.sqrt(
            functions.sum((reshaped - functions.mean(reshaped)) ** 2, axis=1))

        weight /= (reshaped_std + self.eps)
        return functions.reshaped(weight, orig_shape)
