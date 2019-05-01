import typing as tp  # NOQA

import numpy

from chainer import backend
from chainer import functions
from chainer import link
from chainer import links
from chainer import types  # NOQA


def masked_softmax(x, attn_score, mask):
    if mask is not None:
        xp = backend.get_array_module(x.array)
        neg = -1e5 if x.dtype == numpy.dtype16 else -1e18
        attn_score = functions.where(mask, attn_score, neg * xp.ones_like(attn_score))
        attn_weights = functions.softmax(attn_score, axis=-1) * mask
    else:
        attn_weights = functions.softmax(attn_score, axis=-1)
    return attn_weights


def normalize(x, axis=-1, eps=1e-5):
    squared_L2norm = functions.sum(x * x, axis=axis, keepdims=True)
    return x / (squared_L2norm + eps)


class NormalizedScoreProj(link.Link):
    def __init__(self, in_size, out_size=None, initialW=None):
        super(NormalizedScoreProjector, self).__init__()
        with self.init_scope():
            self.g = chainer.Parameter(math.sqrt(in_size), shape=(out_size,))
            self.v = chainer.Parameter(initialW, shape=(out_size, in_units))

    def hybrid_forward(self, x, g, v):
        v = v / function.sqrt(functions.matmul(v, v, transb=True))
        weight = functions.matmul(g, v)
        out = functions.linear(x, weight, None, n_batch_axes=x.ndim - 1)
        return out


class BaseAttention(link.Chain):

    """Base abstract class for attention implementations.

    Extend this class to implement your own attention method. For instance, to define your own `compute_weight()`
    method to calculate the attention weights.
    """

    def __init__(self, *args, **kwargs):
        super(BaseAttention, self).__init__()

    def compute_weight(self, x, query, key, mask=None):
        # type: (types.NdArray, types.NdArray, types.NdArray, tp.Optional[types.NdArray]) -> types.NdArray
        """Compute attention weights based on the query and the keys.

        Args:
            x (:class:`~chainer.Variable` or :ref:`ndarray`):
            query (:class:`~chainer.Variable` or :ref:`ndarray`): The query vectors whose shape is (B, l_q, d_q).
                B, l_q, and d_q represent batch size, query length, and dimension of query.
            key (:class:`~chainer.Variable` or :ref:`ndarray`): The key of the memory whose shape is (B, l_m, d_k).
                B, l_m, and d_k represent batch size, memory length, and dimension of key.
            mask (:class:`~chainer.Variable` or :ref:`ndarray`): If not ``None``, mask the memory slots.
                The shape is (B, l_q, l_m) representing batch size, query length, and memory length.
                Each value is 0 or 1 where 0 means that the memory slot will not be used.

        Returns:
            ~chainer.Variable: Attention weights. The shape is (B, l_q, l_m) and (B, n_head, l_q, l_m) for
                single-head attention and multi-head attention, respectively. n_head is the number of heads.
        """
        raise NotImplementedError

    def read_by_weight(self, x, attn_weights, value):
        # type: (types.NdArray, types.NdArray, types.NdArray) -> types.NdArray
        """Read from the value matrix given the attention weights.

        Args:
            x (:class:`~chainer.Variable` or :ref:`ndarray`):
            attn_weights (:class:`~chainer.Variable` or :ref:`ndarray`): Attention weights.
                The shape is (B, l_q, l_m) and (B, n_head, l_q, l_m) for
                single-head attention and multi-head attention, respectively. n_head is the number of heads.
            value (:class:`~chainer.Variable` or :ref:`ndarray`): The value of the memory whose shape is
                (B, l_m, d_totalValue). B, l_m, and d_totalValue represent batch size, memory length,
                and dimension of total value.

        Returns:
            ~chainer.Variable: Context vector. The shape is (B, l_q, d_contextVector) representing batch size,
                query length, and dimension of context vector.
        """
        output = functions.matmul(attn_weights, value)
        return output

    def forward(self, x , query, key, value=None, mask=None):
        # type: (types.NdArray, types.NdArray, types.NdArray, tp.Optional[types.NdArray], tp.Optional[types.NdArray]) -> tp.Tuple[types.NdArray, types.NdArray]  # NOQA
        """Compute attention weight and context vector.

        Args:
            x (:class:`~chainer.Variable` or :ref:`ndarray`):
            query (:class:`~chainer.Variable` or :ref:`ndarray`): The query vectors whose shape is (B, l_q, d_q).
                B, l_q, and d_q represent batch size, query length, and dimension of query.
            key (:class:`~chainer.Variable` or :ref:`ndarray`): The key of the memory whose shape is (B, l_m, d_k).
                B, l_m, and d_k represent batch size, memory length, and dimension of key.
            value (:class:`~chainer.Variable` or :ref:`ndarray`): The value of the memory whose shape is
                (B, l_m, d_totalValue). B, l_m, and d_totalValue represent batch size, memory length,
                and dimension of total value.
            mask (:class:`~chainer.Variable` or :ref:`ndarray`): If not ``None``, mask the memory slots.
                The shape is (B, l_q, l_m) representing batch size, query length, and memory length.
                Each value is 0 or 1 where 0 means that the memory slot will not be used.

        Returns:
            tuple of :class:`~chainer.Variable`\\s. The first element is context vector and the second is attention
                weights.
        """
        attn_weights = self.compute_weight(x, query, key, mask)
        context_vector = self.read_by_weight(x, attn_weights, value)
        return context_vector, attn_weights


class MultiHeadAttention(BaseAttention):

    """Multi-head Attention.

    In this attention, the input query, key, and value will be applied affine transformation for `n_head` times with
    different weight matrices. Each transformed query, key, and value will be used to calculate the attention weights
    and values. The output of each head will be concatenated to form the final output.

    Args:
        base_cell (BaseAttention): Base attention cell.
        query_size (int): The size of transformed query vector divisible by n_head.
        key_size (int): The size of transformed query vector divisible by n_head.
        value_size (int): The size of transformed query vector divisible by n_head.
        n_head (int): The number of parallel attention heads.
        nobias (bool): If ``True``, use bias in affine transformation of query, key, and value.
        initialW (float, :class:`~chainer.Initializer`, or :ref:`ndarray`): Initializer of weight matrices.
        initial_bias (float, :class:`~chainer.Initializer`, or :ref:`ndarray`): Initializer of bias vectors.
    """

    def __init__(self, base_cell, query_size, key_size, value_size, n_head,
                 nobias=False, initialW=None, initial_bias=None):
        if query_size % n_head != 0:
            raise ValueError(
                'MultiHeadAttention requires `query_size` to be divisible by `n_head`.'
                ' {} % {} = {}'.format(query_size, n_head, query_size % n_head)
            )
        if key_size % n_head != 0:
            raise ValueError(
                'MultiHeadAttention requires `key_size` to be divisible by `n_head`.'
                ' {} % {} = {}'.format(key_size, n_head, query_size % n_head)
            )
        if value_size % n_head != 0:
            raise ValueError(
                'MultiHeadAttention requires `value_size` to be divisible by `n_head`.'
                ' {} % {} = {}'.format(value_size, n_head, query_size % n_head)
            )

        super(MultiHeadAttention, self).__init__()
        self.query_size = query_size
        self.key_size = key_size
        self.value_size = value_size
        self.n_head = n_head
        self.nobias = nobias

        with self.init_scope():
            self.base_cell = base_cell
            projection_kwargs = {'nobias': self.nobias, 'initialW': initialW, 'initial_bias': initial_bias}
            self.project_query = links.Linear(None, self.query_size, **projection_kwargs)
            self.project_key = links.Linear(None, self.key_size, **projection_kwargs)
            self.project_value = links.Linear(None, self.value_size, **projection_kwargs)

    def _project_and_reshape(self, projector, vector, batch_size):
        length = vector.shape[1]
        h = projector(vector, n_batch_axes=2)
        h = functions.reshape(h, (batch_size, length, self.n_head, -1))
        h = functions.transpose(h, (0, 2, 1, 3))
        h = functions.reshape(h, (-1,) + h.shape[2:])
        return h

    def compute_weight(self, x, query, key, mask=None):
        # type: (types.NdArray, types.NdArray, types.NdArray, tp.Optional[types.NdArray]) -> types.NdArray

        batch_size = len(x)
        query = self._project_and_reshape(self.project_query, query, batch_size)
        key = self._project_and_reshape(self.project_key, key, batch_size)
        if mask is not None:
            mask = functions.expand_dims(mask, axis=1)
            mask = functions.repeat(mask, self.n_head, axis=1)
            mask = function.reshape(mask, (batch_size * self.n_head, self.query_size, self.key_size))
        attn_weights = self.base_cell.compute_weight(x, query, key, mask)
        attn_weights = functions.reshape(attn_weights, (batch_size, self.n_head, self.query_size, self.key_size))
        return attn_weights

    def read_by_weight(self, x, attn_weights, value):
        # type: (types.NdArray, types.NdArray, types.NdArray) -> types.NdArray
        attn_weights = functions.reshape(attn_weights, (-1, self.query_size, self.key_size))
        value = self._project_and_reshape(self.project_value, value)
        context_vector = self.base_cell.read_by_weight(x, attn_weights, value)
        context_vector = functions.reshape(context_vector, context_vector.shape[:2] + (self.n_head, -1))
        context_vector = functions.transpose(context_vector, (0, 2, 1, 3))
        context_vector = functions.reshape(context_vector, context_vector.shape[:2] + (-1,))
        return context_vector


class MLPAttention(BaseAttention):

    """Concatenate the query and the key and user a single-hidden-layer MLP to get the attention weights.

    In the standard mode::

    .. math::
        score = v tanh(W[h_q, h_k] + b)

    In the normalized mode:

    .. math::
        score g v / \|v\|_2 tanh(W[h_q, h_k] + b)

    Args:
        h_size (int):
        activation (:class:`~chainer.functions`)
        normalize (bool): If ``True``, normalize the weight that maps the embedded hidden states to the final score.
            This can be considered as a type of Weight Normalization. The default value is ``False``.
        dropout (float): Attention dropout. The default value is 0.0.
        initialW (float, :class:`~chainer.Initializer`, or :ref:`ndarray`)
        initial_bias (float, :class:`~chainer.Initializer`, or :ref:`ndarray`)
    """

    def __init__(self, out_size, activation=function.tanh, normalize=False, dropout=0.0,
                 initialW=None, initial_bias=None):
        super(MLPAttention, self).__init__()
        self.h_size = h_size
        self.activation = activation
        self.normalize = normalize
        self.dropout = dropout

        with self.init_scope():
            self.query_layer = links.Linear(None, self.h_size, initialW=initialW, initial_bias=initial_bias)
            self.key_layer = links.Linear(None, self.h_size, initialW=initialW, initial_bias=initial_bias)

            if not self.normalize:
                self.attn_score = links.Linear(self.h_size, 1, initialW=initialW, initial_bias=initial_bias)
            else:
                self.attn_score = NormalizedLinear(self.h_size, 1, initialW=initialW, initial_bias=initial_bias)

    def compute_weight(self, x, query, key, mask=None):
        # type: (types.NdArray, types.NdArray, types.NdArray, tp.Optional[types.NdArray]) -> types.NdArray
        mapped_query = self.query_layer(query, n_batch_axes=2)
        mapped_key = self.key_layer(key, n_batch_axes=2)
        mid_feature = functions.expand_dims(mapped_query, axis=2) + functions.expand_dims(mapped_key, axis=2)
        mid_feature = self.activation(mid_feature)
        attn_score = self.attn_score(mid_feature)
        attn_weights = functions.dropout(masked_softmax(x, attn_score, mask))
        return attn_weights


class DotProductAttention(BaseAttention):

    """Dot product attention between the qyery and the key.

    Depending on parameters, defined as

    1. ``size`` is ``None``: :math:`score = <h_q, h_k>`
    2. ``size`` is ``None`` and luong style is ``False``: :math:`score = <W_q h_q, W_k h_k>`
    3. ``size`` is ``None`` and luong style is ``True``: :math:`score = <W h_q, h_k>`

    Args:
        size (int): Project the query and the key to vectors with `size` dimension before applying the attention.
            If set to ``None`` (the default value), the query vector and the key vector are directly used to compute
            the attention and should have the same dimension.
        luong_style (bool): The default value is ``False``. If ``True``, the score will be :math:`<W h_q, h_k>`.
            So the size must be the same as the size of the key vector.
        scale (bool): The default value is ``True``. If ``True``, divide the attention weights by the square root of
            the query dimension: :math:`<h_q, h_k> / \sqrt(dim_q)`.
        normalize (bool): The default value is ``False``. If ``True``, the cosine distance is used, i.e.
            :math:`<h_q / \|h_q\|, h_k / \|h_k\|>`.
        nobias (bool): The default value is ``False``. If ``False``, bias is not used in the projection.
        dropout (float): Attention dropout. The default value is 0.0.
        initialW (float, :class:`~chainer.Initializer`, or :ref:`ndarray`)
        initial_bias (float, :class:`~chainer.Initializer`, or :ref:`ndarray`)
    """

    def __init__(self, size=None, luong_style=False, scale=True, normalize=True, nobias=False, dropout=0.0,
                 initialW=None, initial_bias=None):
        if luong_style and size is None:
            raise ValueError('Luong style attention requires `size` to be set explicitly')

        super(DotProductAttention, self).__init__()
        self.size = size
        self.scale = scale
        self.normalize = normalize
        self.nobias = nobias
        self.luong_style = luong_style
        self.dropout = dropout

        if size is not None:
            with self.init_scope():
                project_kwargs = {'nobias': self.nobias, 'initialW': initialW, 'initial_bias': initial_bias}
                self.project_query = links.Linear(None, self.size, **project_kwargs)
                if not self.luong_style:
                    self.project_key = links.Linear(None, self.size, **project_kwargs)
                if self.normalize:
                    self.l2normalize = l2noramzliation(axis=-1)

    def compute_weight(self, x, query, key, mask=None):
        # type: (types.NdArray, types.NdArray, types.NdArray, tp.Optional[types.NdArray]) -> types.NdArray
        if self.size is not None:
            query = self.project_query(query, n_batch_axes=2)
            if not self.luong_style:
                key = self.project_key(key, n_batch_axes=2)
            else:
                if not query.shape[-1] == key.shape[-1]:
                    raise ValueError(
                        'Luong style attention requires key to have the same size as the projected query. ' +
                        'Expect: {}, Actual: {}'.format(key.shape, query.shape))

        if self.normalize:
            query = normalize(query, axis=-1)
            key = normalize(key, axis=-1)
        if self.scale:
            query /= math.sqrt(query.shape[-1])
        attn_score = F.matmul(query, key, transb=True)
        attn_weights = F.dropout(masked_softmax(x, attn_score, mask))
        return attn_weights
