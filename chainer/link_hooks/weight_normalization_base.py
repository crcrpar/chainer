class WeightNormalizationBase(object):

    def __init__(self, weight_name='W', name=None, **kwargs):
        # This attribute represents whether the target link is initialized.
        self.weight_name = weight_name
        if name is not None:
            self.name = name
        self.is_initialized = False

    def __enter__(self):
        raise NotImplementedError(
            '"{}" is not supposed to be used as a context manager.'.format(
                self.name))

    def __exit__(self):
        raise NotImplementedError(
            '"{}" is not supposed to be used as a context manager.'.format(
                self.name))

    def added(self, link):
        if not hasattr(link, self.weight_name):
            raise ValueError(
                'Weight \'{}\' does not exist'.format(self.weight_name))
        if getattr(link, self.weight_name).array is not None:
            self.is_initialized = True
            self.prepare_params(link)

    def forward_preprocess(self, cb_args):
        if not self.is_initialized:
            # We need to initialize the link before `link.forward`
            # that basically initializes the link is called.
            self.initialize_link(cb_args)
            self.prepare_params(cb_args.link)
        # Normalize the target weight and set the normalized as
        # the target link's attribute
        normalized_weight = self.normalize_weight(cb_args)
        setattr(cb_args.link, self.weight_name, normalized_weight)

    def forward_postprocess(self, cb_args):
        setattr(cb_args.link, self.weight_name, self.original_weight)

    def normalize_weight(self, cb_args):
        # 1. Extract weight `W` :class:`~chainer.Parameter`.
        # 2. Hold `W` as this hook's attribute for :meth:`forward_postprocess`
        #    and serialization.
        # 3. Normalize and returns `W`. The return value should be
        #    :class:`~chainer.Variable`
        raise NotImplementedError()

    def initialize_link(self, cb_args):
        link = cb_args.link
        inputs = cb_args.args
        if not hasattr(link, '_initialize_params'):
            raise ValueError('Link cannot be initialized.')
        x = inputs[0]
        link._initialize_params(x.shape[1])

        self.is_initialized = True

    def prepare_params(self, link):
        # This method is for normalizations with states.
        pass
