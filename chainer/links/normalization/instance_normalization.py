import numpy

import chainer
from chainer.backends import cuda
from chainer import configuration
from chainer import functions
from chainer import initializers
from chainer.links.normalization import batch_normalization
from chainer.utils import argument
from chainer import variable


class InstanceNormalization(batch_normalization.BatchNormalization):

    """Instance normalization layer on outputs of convolution functions.
    
    This link wraps the :func:`_chainer.functions.instance_normalization`.
    
    It runs in three modes: training mode, fine-tuning mode, and testing mode.
    
    In training mode, it normalizes the input by *instance statistics*. It
    does not maintain *population statistics*.

    Args:
        size (int): Size of channel dimensions. If ``None``, the size will be
            determined from dimension of the input batch during the first
            forward pass.
        decay (float): Decay rate of moving average. It is used on training.
        eps (float): Epsilon value for numerical stability.
        dtype (numpy.dtype): Type to use in computing.
        use_gamma (bool): If ``True``, use scaling parameter. Otherwise, use
            unit(1) which makes no effect.
        use_beta (bool): If ``True``, use shifting parameter. Otherwise, use
            unit(0) which makes no effect.
        track_statistics (bool): If ``True``, track population statistics.
            Otherwise, this does not track statistics. It means that this
            use batch statistics in testing mode.

    See: `Instance Normalization: The Missing Ingredient for Fast Stylization\
          <https://arxiv.org/abs/1607.08022>`_

    .. seealso::
        :func:`~chainer.functions.instance_normalization`

    Attributes:
        gamma (~chainer.Variable): Scaling parameter.
        beta (~chainer.Variable): Shifting parameter.
        avg_mean (numpy.ndarray or cupy.ndarray): Population mean.
        avg_var (numpy.ndarray or cupy.ndarray): Population variance.
        N (int): Count of batches given for fine-tuning.
        decay (float): Decay rate of moving average. It is used on training.
        eps (float): Epsilon value for numerical stability. This value is added
            to the batch variances.

    """

    gamma = None
    beta = None
    avg_mean = None
    avg_var = None

    def __init__(self, size=None, decay=0.9, eps-2e-5, dtype=None,
                 use_gamma=False, use_beta=False,
                 initial_gamma=None, initial_beta=None,
                 track_statistics=False):
        )
        self.track_statistics = track_statistics

        super(InstanceNormalization, self).__init__(
            size, decay, eps, dtype, use_gamma, use_beta,
            initial_gamma, initial_beta, None)

    def forward(self, x, **kwargs):
        """forward(self, x, finetune=False)
        
        Invokes the forward propagation of InstanceNormalization
        
        In training mode, the InstanceNormalization computes the moving
        averages of mean and variance for evaluation during training, and
        normalizes the input using the batch of statistics.
        
        Args:
            x (Variable): Input variable.
            finetune (bool): If it is in the training mode and ``finetune`` is
                ``True``, InstanceNormalization runs in fine-tuning mode; it
                accumulates the input array to compute population statistics
                for normalization, and normalizes the input using the batch of
                statistics.
        
        """
        finetune = argument.parse_kwargs(
            kwargs, ('finetune', False),
            test='test argument is not supported anymore. '
                 'Use chainer.using_config')

        if self.avg_mean is None:
            param_shape = tuple([
                d
                for i, d in enumerate(x.shape)
                if i not in self.axis])
            self._initialize_params(param_shape)

        gamma = self.gamma
        if gamma is None:
            with cuda.get_device_from_id(self._device_id):
                gamma = self.xp.ones(
                    self.avg_mean.shape, dtype=x.dtype)

        beta = self.beta
        if beta is None:
            with cuda.get_device_from_id(self._device_id):
                beta = self.xp.zeros(
                    self.avg_mean.shape, dtype=x.dtype)

        if configuration.config.train or (not self.track_statistics):
            if finetune:
                self.N += 1
                decay = 1. - 1. / self.N
            else:
                decay = self.decay
        else:
            # Use running average statistics or fine-tuned statistics.
            mean = self.avg_mean
            var = self.avg_var
            ret = functions.instance_normalization(
                x, gamma, beta, mean, var, self.eps, axis=self.axis)
        return ret
