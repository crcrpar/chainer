import typing as tp  # NOQA

import numpy

from chainer import backend
from chainer.functions.array import concat
from chainer.funcitons.array import expand_dims
from chainer.functions.array import split_axis
from chainer.functions.array import repeat
from chainer.functions.array import reshape
from chainer.functions.array import transpose
from chainer.functions.array import where
from chainer.functions.math import matmul
from chainer.functions.noise import dropout
from chainer.functions.connection import linear
from chainer import types  # NOQA
from chainer import variable


InputType = tp.Union[variable.Variable, types.NdArray]


def masked_softmax(
        x,  # type: variable.Variable
        mask=None  # type: tp.Optional[InputType]
    ):
    # type: (...) -> variable.Variable

    if mask is not None:
        xp = backend.get_array_module(x.array)
        neg = -1e5 if x.dtype == numpy.float16 else -1e18
        x = where.where(mask, x, neg * xp.ones_like(x))
        return softmax.softmax(x, axis=-1)
    return softmax.softmax(x, axis=-1)


def multihead_attention(
    n_head,  # type: int
    expected_embed_size,  # type: int
    query,  # type: InputType
    key,  # type: InputType
    value,  # type: InputType
    proj_in_W,  # type: variable.Variable
    proj_in_b,  # type: variable.Variable
    bias_k,  # type: variable.Variable
    bias_v,  # type: variable.Variable
    proj_out_W,  # type: variable.Variable
    proj_out_b,  # type: variable.Variable
    add_zero_attn,  # type: bool
    dropout,  # type: float
    scaling=None,  # typeL tp.Optional[float]
    key_padding_mask=None,  # type: tp.Optional[InputType]
    attn_mask=None,  # type: tp.Optional[InputType]
):
    # type: (...) -> tp.Union[variable.Variable, tp.Tuple[variable.Variable, variable.Variable]]
    """Multi-head Attention forward function.

    Args:
        query (:class:`~chainer.Variable` or :ref:`ndarray`):
            A batch of query vectors whose shape is :math:`(L, B, E)` where
            :math:`L` is the target sequence length, :math:`B` is the
            batch size, and :math:`E` is the embedding size.
        key (:class:`~chainer.Variable` or :ref:`ndarray`)
            A batch of key vectors whose shape is :math:`(S, B, E)` where
            :math:`S` is the source sequence length, :math:`B` is the
            batch size, and :math:`E` is the embedding size.
        value (:class:`~chainer.Variable` or :ref:`ndarray`)
            A batch of value vectors whose shape is :math:`(S, B, E)` where
            :math:`S` is the source sequence length, :math:`B` is the
            batch size, and :math:`E` is the embedding size.
        expected_embedding_size (int): Total number of units of the model.
        n_head (int): The number of parallel attention heads.
        proj_in_W (:class:`~chainer.Variable` or :ref:`ndarray`:)
        proj_in_b (:class:`~chainer.Variable` or :ref:`ndarray`:)
        add_zero_attn (bool): If ``True``, add a new batch of zeros to
            the key and value sequences at axis=1.
        dropout (float): Dropout ratio.
        proj_out_W (:class:`~chainer.Variable` or :ref:`ndarray`)
        proj_out_b (:class:`~chainer.Variable` or :ref:`ndarray`)
        key_padding_mask (:class:`~chainer.Variable` or :ref:`ndarray`):
            If not ``None``, specified padding elements in the key
            will be ignored by the attention.
            The shape is :math:`(B, S)` where :math:`B` is the batch size,
            and :math:`S` is the source sequence length.
        attn_mask (:class:`~chainer.Variable` or :ref:`ndarray`):
            Mask help attention ignores certain positions.
            The shape is :math:`(L, L)` where :math:`L` is
            the target sequence length.

    Outputs:
        tuple of :class:`~chainer.Variable`\\ s.
        - The first element is the output of attention with the shape of
        :math:`(L, B, E)` where :math:`L` is the target sequence length,
        :math:`B` is the batch size, and :math:`E` is the embedding size.
        - The second element is the weights of the attention with the shape of
        :math:`(B, L, S)` where :math:`B` is the batch size, :math:`L` is the 
        target sequence length, and :math:`S` is the source sequence length.
    """

    def _in_proj(x, W, b=None, start=0, end=None):
        W = W[start:end, :]
        if b is not None:
            b = b[start:end]
        return linear.linear(x, W, b, n_batch_axes=x.ndim-1)

    def _in_proj_qkv(query, W, b):
        return split_axis.split_axis(_in_proj(query, W, b), 3, axis=-1)

    def _in_proj_kv(key, W, b, embedding_size):
        return split_axis.split_axis(_in_proj(key, W, b, embedding_size), 2, axis=-1)

    def _in_proj_q(query, W, b, embedding_size):
        return _in_proj(query, W, b, embedding_size)

    def _in_proj_k(key, W, b, embedding_size):
        return _in_proj(value, W, b, embedding_size, 2 * embedding_size)

    def _in_proj_v(value, W, b, embedding_size):
        return _in_proj(value, W, b, 2 * embedding_size)

    if embedding_size % n_head != 0:
        raise ValueError(
            '`embedding_size` ({}) need to be divisible by `n_head` ({})'.format(embedding_size, n_head))
    if (bias_k is None) != (bias_v is None):
        raise ValueError
    qkv_same = (query is key) and (query is value)
    kv_same = key is value
    target_length, batch_size, embedding_size = query.shape
    head_size = embedding_size // n_head
    if scaling is None:
        scaling = head_size ** -0.5

    xp = backend.get_array_module(query)
    dtype = query.dtype

    if qkv_same:
        # self-attention
        q, k, v = _in_proj_qkv(query, proj_in_W, pro_in_b)
    elif kv_same:
        q = _in_proj_q(query, proj_in_W, proj_in_b, embedding_size)
        if key is None:
            k, v = None, None
        else:
            k, v = _in_proj_kv(key, proj_in_W, proj_in_b, embedding_size)
    else:
        q = _in_proj_q(query, proj_in_W, proj_in_b, embedding_size)
        k = _in_proj_k(key, proj_in_W, proj_in_b, embedding_size)
        v = _in_proj_v(value, proj_in_W, proj_in_b, embedding_size)
    q *= scaling

    if bias_k is not None:
        k = concat.concat((k, repeat.repeat(bias_k, batch_size, axis=1)))
        v = concat.concat((v, repeat.repeat(bias_v, batch_size, axis=1)))
        if attn_mask is not None:
            attn_mask = concat.concat(
                (
                    attn_mask,
                    xp.zeros((len(attn_mask), 1), dtype=dtype)
                )
            )
        if key_padding_mask is not None:
            key_padding_mask = concat.concat(
                (
                    key_padding_mask,
                    xp.zeros((len(key_padding_mask), 1), dtype=dtype)
                )
            )
    q = reshape.reshape(q, (target_length, batch_size * n_head, head_size))
    q = transpose.transpose(q, (1, 0, 2))
    if k is not None:
        k = reshape.reshape(k, (-1, batch_size * n_head, head_size))
        k = transpose.transpose(k, (1, 0, 2))
        v = reshape.reshape(v, (-1, batch_size * n_head, head_size))
        v = transpose.transpose(v, (1, 0, 2))

    # TODO(crcrpar): Investigate the possibility that
    # the input of `key` is `None` and `bias_k` is also `None`
    source_length = k.shape[1]

    if key_padding_mask is not None:
        if key_padding_mask.shape[:2] != (batch_size, source_length):
            raise ValueError('`key_padding_mask` has wrong shape. Expected: ({}, {}), Actual: ({}, {})'.format(batch_size, source_length, key_padding_mask.shape[0], key_padding_mask.shape[1]))

    if add_zero_attn:
        source_length += 1
        k = concat.concat((k, xp.zeros((len(k), 1), dtype=dtype)))
        v = concat.concat((v, xp.zeros((len(v), 1), dtype=dtype)))
        if attn_mask is not None:
            attn_mask = concat.cocnat(
                (attn_mask, xp.zeros((len(attn_mask), 1), dtype=dtype)))
        if key_padding_mask is not None:
            key_padding_mask = concat.concat(
                (
                    key_padding_mask,
                    xp.zeros(
                        (len(key_padding_mask), 1), key_padding_mask.dtype)
                )
            )
    attn_output_weights = matmul.matmul(
        q,
        transpose.transpose(k, (0,) + (2, 1) + tuple(range(2, k.ndim))))
    if attn_output_weights.shape != (batch_size * n_head, target_length, source_length):
        raise ValueError('`attn_output_weights` is shaped wrongly')

    if attn_mask is not None:
        attn_mask = expand_dims.expand_dims(attn_mask, 0)
        attn_output_weights += attn_mask

    if key_padding_mask is not None:
        attn_output_weights = reshape.reshape(
            attn_output_weights,
            (batch_Size, n_head, target_length, source_length)
        )
        attn_output_weights -= xp.inf * expand_dims.expand_dims(expand_dims.expand_dims(key_padding_mask, 1), 2)
        attn_output_weights = reshape.reshape(
            attn_output_weights,
            (batch_size * n_head, target_length, source_length)
        )

    attn_output_weights = softmax.softmax(attn_output_weights, axis=-1)
    attn_output_weights = dropout.dropout(attn_output_weights, dropout)

    attn_output = matmul.matmul(attn_output_weights, v)
    attn_output = transpose.transpose(
        attn_output,
        (1, 0) + tuple(range(2, attn_output.ndim))
    )
    attn_output = reshape.reshape(
        attn_output, (target_length, batch_size, embedding_size))
    attn_output = linear.linear(
        attn_output, proj_out_W, proj_out_B,
        n_batch_axes=attn_output.ndim-1)
    return attn_output, attn_output_weights
