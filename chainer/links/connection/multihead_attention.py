import typing as tp  # NOQA


from chainer import functions
from chainer import initializers
from chainer import link
from chainer import links
from chainer import types  # NOQA
from chainer import variable


InputType = tp.Union[variable.Variable, types.NdArray]


class MultiHeadAttention(link.Chain):

    """Multi-Head Attention.

    Args:
        n_head (int): The number of heads.
        embed_size (int):
            The size of input query vectors
            and projected query, key, and value vectors.
        self_attention (bool): If ``True``, this becomes self-attention.
        ksize (int):
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

    def __init__(
        self,
        n_head,                 # type: int
        embed_size,             # type: int
        self_attention=False,   # type: tp.Optional[bool]
        ksize=None,             # type: tp.Optional[int]
        vsize=None,             # type: tp.Optional[int]
        attention_dropout=0.0,  # type: tp.Optional[float]
        post_dropout=0.0,       # type: tp.Optional[float]
        scaling=None,           # type: tp.Optional[float]
        initialW=None,          # type: tp.Optional[types.InitializerSpec]
        initial_bias=None,      # type: tp.Optional[types.InitializerSpec]
        nobias=False,           # type: tp.Optional[bool]
        nobias_kv=True          # type: tp.Optional[bool]
    ):
        # type (...) -> None
        super().__init__()

        if embed_size % n_head != 0:
            raise ValueError(
                '`embed_size` ({}) must be divisible by `n_head` ({})'.format(
                    embed_size, n_head))
        if not self_attention and (ksize is None or vsize is None):
            raise ValueError(
                '`ksize` and `vsize` are required '
                'if `self_attention` is `False`.')
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
                self.proj_in_W = variable.Parameter(
                    _initialW, (3 * self.embed_size, self.embed_size))  # type: variable.Variable  # NOQA
            else:
                self.proj_q_weight = variable.Parameter(
                    _initialW, (self.embed_size, self.embed_size))  # type: variable.Variable  # NOQA
                self.proj_k_weight = variable.Parameter(
                    _initialW, (self.embed_size, self.ksize))  # type: variable.Variable  # NOQA
                self.proj_v_weight = variable.Parameter(
                    _initialW, (self.embed_size, self.vsize))  # type: variable.Variable  # NOQA
            if not nobias:
                self.proj_in_bias = variable.Parameter(
                    _initial_bias, (3 * self.embed_size,))  # type: variable.Variable  # NOQA
            else:
                self.proj_in_bias = None

            self.out_proj = links.Linear(
                self.embed_size, self.embed_size,
                initialW=_initialW, initial_bias=_initial_bias, nobias=nobias)
            self.proj_out_W = self.out_proj.W
            self.proj_out_b = self.out_proj.b

            if not nobias_kv:
                self.bias_k = variable.Parameter(
                    _initial_bias, (self.embed_size,))  # type: variable.Variable  # NOQA
                self.bias_v = variable.Parameter(
                    _initial_bias, (self.embed_size,))  # type: variable.Variable  # NOQA
            else:
                self.bias_k, self.bias_v = None, None

    def forward(
        self,
        query,                    # type: tp.Optional[InputType]
        key=None,                 # type: tp.Optional[InputType]
        value=None,               # type: tp.Optional[InputType]
        key_padding_mask=None,    # type: tp.Optional[InputType]
        attention_mask=None,      # type: tp.Optional[InputType]
        add_zero_attention=False, # type: tp.Optional[bool]
        return_weights=False      # type: tp.Optional[bool]
    ):
        # type: (...) -> tp.Union[tp.Tuple[variable.Variable, variable.Variable], variable.Variable]  # NOQA
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
            tuple of :class:`~chainer.Variable`\\s
                if `return_weights` is ``True``.
                Otherwise, :class:`~chainer.Variable`.
                The first element is context vector shaped :math:`(L, B, E)`
                the second is attention weights shaped :math:`(B, L, S)`.
        """
        # TODO (crcrpar): Support cuDNN MultiHeadAttn when CuPy supports it.

        if hasattr(self, 'proj_in_W'):
            proj_in_W = self.proj_in_W
        else:
            proj_in_W = (
                self.proj_q_weight, self.proj_k_weight, self.proj_v_weight)

        attention, attention_weights = functions.multihead_attention(
            self.n_head, self.ebedding_size, query, key, value,
            proj_in_W, self.proj_in_b, self.bias_k, self.bias_v,
            self.proj_out_W, self.proj_out_b,
            add_zero_attention, self.attention_dropout,
            key_padding_mask, attention_mask, self.scaling
        )
        if return_weights:
            return attention, attention_weights
        return attention
