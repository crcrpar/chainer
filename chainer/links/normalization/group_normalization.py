import numpy

from chainer.backends import cuda
from chainer import configuration
from chainer import functions
from chainer.functions.array.hstack import hstack
from chainer.links.normalization.batch_normalization import BatchNormalization
from chainer.utils import argument
from chainer import variable


class GroupNormalization(BatchNormalization):

    """Group normalization layer on outputs of liner or convolution functions.

    This links wraps the :func:`~chainer.links.BatchNormalization`.

    Args:
        ngroups (int): Number of channel groups.
        nchannels (int): Number of input arrays channels. This argument is
            the counterpart of BatchNormalization size.
        eps (float): Epsilon value for numerical stability.
        use_gamma (bool): If ``True``, use scaling parameter. Otherwise, use
            unit(1) which makes no effect.
        use_beta (bool): If ``True``, use shifting parameter. Otherwise, use
            unit(0) which makes no effect.
        track_statistics (bool): If ``True``, use running mean and average in
            inference. Otherwise, there is no difference between training and

        initial_gamma (~chainer.Initializer): Initializer for scaling
            parameter. If ``None``, the vector is filled by 1. If a scalar,
            the parameter is filled by it. If ``numpy.ndarray``, the parameter
            is set by it.
        initial_beta (~chainer.Initializer): Initializer for shifting
            parameter. If ``None``, the vector is filled by 1. If a scalar,
            the parameter is filled by it. If ``numpy.ndarray``, the parameter
            is set by it.

    """  # NOQA

    def __init__(self, ngroups, nchannels, eps=1e-5, dtype=numpy.float32,
                 use_gamma=True, use_beta=True, track_statistics=False,
                 initial_gamma=None, initial_beta=None):
        if nchannels % ngroups != 0:
            raise ValueError("Invalid ngroups for nchannels.")
        super(GroupNormalization, self).__init__(
            size=ngroups, eps=eps, dtype=dtype, use_gamma=use_gamma,
            use_beta=use_beta, initial_gamma=initial_gamma,
            initial_beta=initial_beta)
        self.ngroups = ngroups
        self.track_statistics = track_statistics

    def __call__(self, x, **kwargs):
        """Apply Group normalization to given input.

        Args:
            x (~chainer.Variable): Input variable.
            finetune (bool): If it is in the training mode and ``finetune`` is
                ``True``, GroupNormalization runs in fine-tuning mode; it
                accumulates the input array to compute population statistics
                for normalization, and normalizes the input using batch
                statistics.
        """
        argument.check_unexpected_kwargs(
            kwargs, test='test argument is not supported anymore. '
            'Use chainer.using_config')
        finetune, = argument.parse_kwargs(kwargs, ('finetune', False))

        org_shape = x.shape
        x = self.xp.reshape(x, (1, org_shape[0] * self.ngroups, -1))
        if hasattr(self, 'gamma'):
            gamma = self.gamma
        else:
            with cuda.get_device_from_id(self._device_id):
                gamma = variable.Variable(self.xp.ones(
                    self.ngroups, dtype=x.dtype))

        if hasattr(self, 'beta'):
            beta = self.beta
        else:
            with cuda.get_device_from_id(self._device_id):
                beta = variable.Variable(self.xp.zeros(
                    self.ngroups, dtype=x.dtype))
        gamma = hstack([gamma] * org_shape[0])
        beta = hstack([beta] * org_shape[0])

        if configuration.config.train:
            if finetune and self.track_statistics:
                self.N += 1
                decay = 1. - 1. / self.N
            else:
                decay = self.decay

            ret = functions.batch_normalization(
                x, gamma, beta, eps=self.eps,
                running_mean=self.avg_mean, running_var=self.avg_var,
                decay=decay)
        elif self.track_statistics:
            mean = variable.Variable(self.avg_mean)
            var = variable.Variable(self.avg_var)
            ret = functions.fixed_batch_normalization(
                x, gamma, beta, mean, var, self.eps)
        else:
            decay = self.decay
            ret = functions.batch_normalization(
                x, gamma, beta, eps=self.eps,
                running_mean=self.avg_mean, running_var=self.avg_var,
                decay=decay)
        ret = self.xp.reshape(ret, org_shape)
        return ret
