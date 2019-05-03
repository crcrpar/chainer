import unittest

import numpy
import pytest

import chainer
from chainer import functions
from chainer import links
from chainer import testing


def _matmul(x, y, transa=False, transb=False):
    with chainer.no_backprop_mode(), \
            chainer.using_config('use_cuda', False), \
            chainer.using_config('use_chainerx', False):
        out = functions.matmul(x, y, transa=transa, transb=transb).array
    return out


def _softmax(x, axis=-1):
    with chainer.no_backprop_mode(), \
            chainer.using_config('use_cuda', False), \
            chainer.using_config('use_chainerx', False):
        y = functions.softmax(x, axis=axis).array
    return y


def _scaled_dot_attention(q, k, v, shape, unseen_mask=False, source_lengths=None):
    qkt = _matmul(q, numpy.transpose(k, (0, 1, 3, 2)) / numpy.sqrt(shape[3], dtype=q.dtype))

    if unseen_mask or source_lengths is not None:
        b1, b2, s1, s2 = qkt.shape
        for i in range(b1):
            for j in range(b2):
                for m in range(s1):
                    for n in range(s2):
                        if unseen_mask or n > m:
                            qkt[i, j, m, n] = -numpy.inf
                        if (source_lengths is not None
                                and n >= source_lengths[i]):
                            qkt[i, j, m, n] = -numpy.inf
    ref = _softmax(qkt)
    ref = _matmul(ref, v)
    return ref


def _generate_source_lengths(batch_size, sequence_length):
    source_lengths = numpy.random.randint(1, sequence_length + 1, (batch_size,))
    # max source length has to be equal to seqence_length, so that
    # randomly choose one example to have source_length = sequence_length
    source_lengths[numpy.random.randint(batch_size)] = sequence_length
    source_length_tensor = source_lengths.astype(numpy.int32)
    return source_lengths, source_length_tensor


def _split_heads_ref(x, shape, n_head, d_head):
    x_split = numpy.reshape(x, shape[:2] + (n_head, d_head))
    x_split_T = numpy.transpose(x_split, (0, 2, 1, 3))
    ref = numpy.reshape(x_split_T, (shape[0], n_head, shape[1], d_head))
    return ref


def _combine_heads_ref(x, shape, n_head, d_head):
    x_T = numpy.transpose(x, (0, 2, 1, 3))
    ref = numpy.reshape(x_T, shape[:2] + (n_head * d_head,))
    return ref


def _create_source_lengths_mask(batch_size, source_lengths):
    max_source_length = numpy.max(source_lengths)
    source_indices = numpy.arange(
        max_source_length)[None, Ellipsis].astype(source_lengths.dtype)
    source_indices = numpy.broadcast(
        source_indices, (batch_size, max_source_length))
    source_lengths = numpy.broadcast(
        source_lengths[Ellipsis, None], (batch_size, max_source_length))
    return source_indices < source_lengths


def _fc(x, x_name, model, start=None, end=None):
    x_fc_W, x_fc_b = None, None
    for name, param in model.named_params():
        if x_name + 'W' in name:
            assert x_fc_W is None
            x_fc_W = param.array[start:end, :]
        if x_name + 'b' in name:
            assert x_fc_b is None
            x_fc_b = param.array[start:end]
    return numpy.matmul(x, x_fc_W.T) + x_fc_b


@testing.parameterize(*testing.product({
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'embed_size': [5],
    'self_attention': [True, False],
    'n_head': [1, 4],
    'qsize': [10],
    'ksize': [12],
    'vsize': [14],
    'attention_dropout': [0.0, 0.5],
    'post_dropout': [0.0, 0.5],
    'scaling': [None, 1.0],
    'initialW': [None],
    'initial_bias': [None],
    'nobias': [True, False],
    'nobias_kv': [True, False],
    'use_source_lengths': [True, False],
}))
class TestMultiHeadAttention(unittest.TestCase):

    def setUp(self):
        self.link = links.MultiHeadAttention(
            self.n_head, self.embed_size, self.self_attention,
            self.ksize, self.vsize,
            self.attention_dropout, self.post_dropout, self.scaling,
            self.initialW, self.initial_bias, self.nobias, self.nobias_kv)
        for param in self.link.params():
            param.array[:] = param.array.astype(self.dtype)

    def test_forward(self):
        batch_size, sequence_length = numpy.random.randint(2, 11, (2,))
        d_head, n_head = numpy.random.randint(3, 11)
        d_model = d_head * n_head
        shape = [batch_size, sequence_length, d_model]

        source_lengths = None
        source_lengths_tensor = None
        if self.use_source_lengths:
            source_lengths, source_lengths_tensor = _generate_source_lengths(
                batch_size, sequence_length)

        decoder_state = numpy.random.rand(
            batch_size, d_model).astype(self.dtype)
        k = numpy.random.rand(*shape).astype(self.dtype)
        v = k
        q = numpy.expand_dims(decoder_state, 1)
        decoder_state_tensor = numpy.copy(decoder_state)
        source_hidden_tensor = numpy.transpose(
            k, (1, 0) + tuple(range(2, len(shape))))

        _bs = len(decoder_state_tensor)
        _q = numpy.expand_dims(q, 1)
        _q = numpy.transpose(_q, (1, 0) + tuple(range(2, _q.ndim)))
        _k = source_hidden_tensor
        _v = source_hidden_tensor
        source_length_mask = None
        if source_lengths is not None and self.use_source_lengths:
            source_length_mask_int = _create_source_lengths_mask(
                _bs, source_lengths_tensor)
            source_length_mask = source_length_mask != 1

        result = self.link(_q, _k, _v, key_padding_mask=source_length_mask,
                           return_weights=False)

        q_fc = _fc(_q, "proj_in_", self.link, end=d_model)
        k_fc = _fc(_k, "proj_in_", self.link, start=d_model, end=2 * d_model)
        v_fc = _fc(_v, "proj_in_", self.link, start=2 * d_model)
        q_split = _split_heads_ref(
            q_fc, (batch_size, 1, d_model), n_head, d_head)
        k_split = _split_heads_ref(k_fc, shape, n_head, d_head)
        v_split = _split_heads_ref(v_fc, shape, n_head, d_head)

        attention_heads = _scaled_dot_attention(
            q_split, k_split, v_split, shape, source_lengths=source_lengths)

        combined_attention_heads = _combine_heads_ref(
            attention_heads, (batch_size, 1), n_head, d_head)
        reference = _fc(combined_attention_heads, "proj_out.", self.link)
        reference = numpy.squeeze(reference, axis=1)
