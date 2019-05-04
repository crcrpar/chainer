import typing as tp  # NOQA

import numpy

from chainer import backend
from chainer import functions
from chainer import initializers
from chainer import link
from chainer import links
from chainer import types  # NOQA
from chainer import variable


def masked_softmax(x, mask=None):
    # type: (variable.Variable, tp.Optional[variable.Variable]) -> variable.Variable  # NOQA
    if mask is not None:
        xp = backend.get_array_module(x.array)
        neg = -1e5 if x.dtype == numpy.dtype16 else -1e18
        x = functions.where(
            mask, x, neg * xp.ones_like(x))
        y = functions.softmax(x, axis=-1) * mask
    else:
        y = functions.softmax(x, axis=-1)
    return y


class MultiHeadAttention(link.Chain):

    """Multi-Head Attention.

    Args:
        n_head (int): The number of heads.
        embed_size (int):
            The size of input query vectors
            and projected query, key, and value vectors.
        self_attention (bool): If ``True``, this becomes self-attention.
        kize (int):
            The size of input key vectors.
        vsize (int):
            The size of input value vectors.
        attention_dropout (float):
            The dropout ratio applied to attention before softmax.
        post_dropout (float):
            The dropout ratio applied to attention after softmax.
        scaling (float):
            The scaler value that defaults to :math:`1/\sqrt(n_{head})`.
        initialW (:ref:`initializer <initializer>`): Initializer to initialize
            the weight.
        initial_bias (:ref:`initializer <initializer>`): Initializer to
            initialize the bias. If ``None``, the bias will be initialized to
            zero.
        nobias (bool): Whether to add bias to projected query, key, and value
            if the instance of this class is used as self-attention.
        nobias_kv (bool):
            If ``True``, no bias is added to projected key and value.

    See: `Attention Is All You Need<https://arxiv.org/abs/1706.03762>`_
    """

    def __init__(self, n_head, embed_size, self_attention=False,
                 ksize=None, vsize=None,
                 attention_dropout=0.0, post_dropout=0.0, scaling=None,
                 initialW=None, initial_bias=None,
                 nobias=False, nobias_kv=True):
        # type (int, int, tp.Optional[bool], tp.Optional[int], tp.Optional[int], tp.Optional[float], tp.Optional[float], tp.Optional[float], tp.Optional[types.InitializerSpec], tp.Optional[types.InitializerSpec], tp.Optional[bool], tp.Optional[bool]) -> 'MultiHeadAttention'  # NOQA
        super().__init__()

        if embed_size % n_head != 0:
            raise ValueError('`embed_size` must be divisible by `n_head`')
        if not self_attention and (ksize is None or vsize is None):
            raise ValueError('`ksize` and `vsize` are necessary.')
        else:
            ksize = embed_size
            vsize = embed_size
        self.n_head = n_head
        self.embed_size = embed_size  # == qsize
        self.head_size = self.embed_size // self.n_head
        self._self_attention = self_attention
        if scaling is None:
            self.scaling = self.head_size ** -0.5
        else:
            self.scaling = scaling

        if self._self_attention:
            ksize = self.embed_size
            vsize = self.embed_size
        self.ksize = ksize
        self.vsize = vsize
        self.qkv_same_size = (
            self.embed_size == self.ksize and self.embed_size == self.vsize)

        self.attention_dropout = attention_dropout
        self.post_dropout = post_dropout

        with self.init_scope():
            if initialW is None:
                _initialW = initializers.GlorotNormal()
            if initial_bias is None:
                _initial_bias = initializers.Zero()
            if self.qkv_same_size:
                self.proj_in_weight = variable.Parameter(
                    _initialW, (3 * self.embed_size, self.embed_size))  # type: variable.Variable  # NOQA
            else:
                self.proj_k_weight = variable.Parameter(
                    _initialW, (self.embed_size, self.ksize))  # type: variable.Variable  # NOQA
                self.proj_v_weight = variable.Parameter(
                    _initialW, (self.embed_size, self.vsize))  # type: variable.Variable  # NOQA
                self.proj_q_weight = variable.Parameter(
                    _initialW, (self.embed_size, self.embed_size))  # type: variable.Variable  # NOQA
            if not nobias:
                self.proj_in_bias = variable.Parameter(
                    _initial_bias, (3 * self.embed_size,))  # type: variable.Variable  # NOQA
            else:
                self.proj_in_bias = None

            self.out_proj = links.Linear(
                self.embed_size, self.embed_size,
                initialW=_initialW, initial_bias=_initial_bias, nobias=nobias)

            if not nobias_kv:
                self.bias_k = variable.Parameter(
                    _initial_bias, (self.embed_size,))  # type: variable.Variable  # NOQA
                self.bias_v = variable.Parameter(
                    _initial_bias, (self.embed_size,))  # type: variable.Variable  # NOQA
            else:
                self.bias_k, self.bias_v = None, None

    def proj_in(self, x, start_idx=0, end_idx=None):
        # type: (variable.Variable, tp.Optional[int], tp.Optional[int]) -> variable.Variable  # NOQA
        W = self.proj_in_weight[start_idx:end_idx, :]
        b = self.proj_in_bias
        if b is not None:
            b = b[start_idx:end_idx]
        return functions.linear(x, W, b, n_batch_axes=x.ndim - 1)

    def proj_in_qkv(self, query):
        # type: (variable.Variable) -> variable.Variable  # NOQA
        return functions.split_axis(self.proj_in(query), 3, axis=-1)

    def proj_in_query(self, query):
        # type: (variable.Variable) -> variable.Variable  # NOQA
        if self.qkv_same_size:
            return self.proj_in(query, end=self.embed_size)
        else:
            bias = self.proj_in_bias
            if bias is not None:
                bias = bias[:self.embed_size]
            return functions.linear(
                query, self.proj_q_weight, bias, n_batch_axes=query.ndim - 1)

    def proj_in_key(self, key):
        # type: (variable.Variable) -> variable.Variable  # NOQA
        if self.qkv_same_dim:
            return self.proj_in(
                key, start=self.embed_size, end=2 * self.embed_size)
        else:
            bias = self.proj_in_bias
            if bias is not None:
                bias = bias[self.embed_size:2 * self.embed_size]
            return functions.linear(
                key, self.proj_k_weight, bias, n_batch_axes=key.ndim - 1)

    def proj_in_v(self, value):
        # type: (variable.Variable) -> variable.Variable  # NOQA
        if self.qkv_same_size:
            return self.proj_in(value, start=2 * self.embed_size)
        else:
            bias = self.proj_in_bias
            if bias is not None:
                bias = bias[2 * self.embed_size:]
            return functions.linear(
                value, self.proj_v_weight, bias, n_batch_axes=value.ndim - 1)

    def forward(self, query, key=None, value=None, key_padding_mask=None,
                attention_mask=None, return_weights=False):
        # type: (variable.Variable, tp.Optional[variable.Variable], tp.Optional[variable.Variable], tp.Optional[variable.Variable], tp.Optional[bool]) -> tp.Union[tp.Tuple[variable.Variable, variable.Variable], variable.Variable]  # NOQA
        """Compute attention weight and context vector.

        Self-attention can be implemented by passing the same arguments for
        query, key, and value. Timesteps can be masked by
        giving a time x time mask in the `attention_mask`. Padding elements
        can be excluded from the key by passing a batch_size x source_length
        mask where padding elements are indicated by 1.

        Args:
            query (:class:`~chainer.Variable` or :ref:`ndarray`):
                The query vectors with the shape of
                (time, batch_size, query_in_size).
            key (:class:`~chainer.Variable` or :ref:`ndarray`):
                The key of the memory vectors with the shape of
                (time, batch_size, key_in_size).
            value (:class:`~chainer.Variable` or :ref:`ndarray`):
                The value of the memory with the shape of
                (time, batch_size, value_in_size).
            key_padding_mask (:class:`~chainer.Variable` or :ref:`ndarray`):
                If not ``None``, mask the memory slots.
                The shape is (batch_size, source_length).
                Each value is 0 or 1 where 1 means that
                the memory slot will not be used.
            attention_mask (:class:`~chainer.Variable` or :ref:`ndarray`)
            return_weights (bool):
                If ``True``, return both attention and attention weights.
        Returns:
            tuple of :class:`~chainer.Variable`\\s.
                The first element is context vector and
                the second is attention weights.
        """

        # TODO (crcrpar): Support cuDNN MultiHeadAttn when CuPy supports it.

        if self._self_attention:
            if key is None:
                key = query
            if value is None:
                value = query

        qkv_same = (query is key) and (query is value)
        if self._self_attention and not qkv_same:
            raise ValueError(
                '`MultiHeadAttention` is initialized as self-attention, '
                'but the input query, key, and value are not identical.')
        kv_same = key is value
        target_length, batch_size, embed_size = query.shape

        if self.qkv_same_size:
            q, k, v = self.proj_in_qkv(query)
        assert embed_size == self.embed_size, \
            'Expected `embed_size`: {}, Actual: {}'.format(
                self.embed_size, embed_size)

        if self.qkv_same_size:
            q, k, v = self.proj_in_qkv(query)
        elif kv_same:
            q = self.proj_in_q(query)
            if key is None:
                assert value is None
            else:
                k = self.proj_in_k(key)
                v = self.proj_in_v(key)
        else:
            q = self.proj_in_q(query)
            k = self.proj_in_q(key)
            v = self.proj_in_q(value)

        q *= self.scaling

        if self.bias_k is not None:
            k = functions.concat(
                (k, functions.repeat(self.bias_k, 1, batch_size)))
            v = functions.concat(
                (v, functions.repeat(self.bias_v, 1, batch_size)))
            if attention_mask is not None:
                attention_mask = functions.concat(
                    (attention_mask, self.xp.zeros((len(attention_mask), 1),
                    dtype=attention_mask.dtype)), axis=1)
            if key_padding_mask is not None:
                key_padding_mask = functions.concat(
                    (
                        key_padding_mask,
                        self.xp.zeros(
                            (len(key_padding_mask), 1),
                            dtype=key_padding_mask.dtype
                        )
                    ), axis=1)
        q = functions.reshape(
            q, (target_length, batch_size * self.n_head, self.head_size))
        q = functions.transpose(q, (1, 0) + tuple(range(2, q.ndim)))
        if k is not None:
            k = functions.reshape(
                k, (-1, batch_size * self.n_head, self.head_size))
            k = functions.transpose(k, (1, 0, 2))
        if v is not None:
            v = functions.reshape(
                v, (-1, batch_size * self.n_head, self.head_size))
            v = functions.transpose(v, (1, 0, 2))

        attention_weights = functions.matmul(
            q, functions.transpose(k, (0, 2, 1)))
        if attention_mask is not None:
            attention_weights += attention_mask

        source_length = k.shape[1]
        if key_padding_mask is not None:
            assert key_padding_mask.shape[:2] == (batch_size, source_length)

        attention_weights = masked_softmax(attention_weights, key_padding_mask)
        attention_weights = functions.dropout(
            attention_weights, self.attention_dropout)
        attention = functions.matmul(attention_weights, v)
        attention = functions.reshape(
            functions.transpose(
                attention, (1, 0) + tuple(range(2, attention.ndim))),
            (target_length, batch_size, embed_size))

        self.prev_attention_weights = attention_weights
        self.prev_attention_weights.unchain()
        if return_weights:
            attention_weights = functions.reshape(
                attention_weights,
                (batch_size, self.n_head, target_length, source_length))
            return attention, functions.mean(attention_weights, axis=1)
        return attention
