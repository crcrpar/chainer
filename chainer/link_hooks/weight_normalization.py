from chainer import link_hook


class WeightNormalizationBase(link_hook.LinkHook):

    name = 'WeightNormalizationBase'

    def __init__(self, *args, **kwargs):
        self.is_initialized = False

    def __enter__(self):
        raise NotImplementedError(
            'This hook is not supposed to be used as context manager.')

    def added(self, link):
        if link.W.array is not None:
            self.is_initialized = True
            self.prepare_params(link)

    def forward_preprocess(self, cb_args):
        if not self.is_initialized:
            self._initialize_link(cb_args)
            self.prepare_params(cb_args.link)

        self._normalize(cb_args)

    def _normalize(self, cb_args):
        with chainer.using_device(cb_args.link.device):
            self.normalize(cb_args)

    def normalize(self, cb_args):
        raise NotImplementedError(
            'WeightNormalizationBase cannot be used directly.')

    def initialize_link(self, cb_args):
        link = cb_args.link
        inputs = cb_args.args
        if not hasattr(link, '_initialize_params'):
            raise RuntimeError('Link cannot be initialized.')
        link._initialize_params(inputs[0].shape[1])

        self.is_initialized = True

    def prepare_params(self, link):
        # Declare parameters and buffers necessary for this hook.
        raise NotImplementedError(
            'WeightNormalizationBase cannot be used directly.')
