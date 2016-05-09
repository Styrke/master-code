# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Module for constructing RNN Cells."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import control_flow_ops# import identity

from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh


class RNNCell(object):
  """Abstract object representing an RNN cell.

  An RNN cell, in the most abstract setting, is anything that has
  a state -- a vector of floats of size self.state_size -- and performs some
  operation that takes inputs of size self.input_size. This operation
  results in an output of size self.output_size and a new state.

  This module provides a number of basic commonly used RNN cells, such as
  LSTM (Long Short Term Memory) or GRU (Gated Recurrent Unit), and a number
  of operators that allow add dropouts, projections, or embeddings for inputs.
  Constructing multi-layer cells is supported by a super-class, MultiRNNCell,
  defined later. Every RNNCell must have the properties below and and
  implement __call__ with the following signature.
  """

  def __call__(self, inputs, state, scope=None):
    """Run this RNN cell on inputs, starting from the given state.

    Args:
      inputs: 2D Tensor with shape [batch_size x self.input_size].
      state: 2D Tensor with shape [batch_size x self.state_size].
      scope: VariableScope for the created subgraph; defaults to class name.

    Returns:
      A pair containing:
      - Output: A 2D Tensor with shape [batch_size x self.output_size]
      - New state: A 2D Tensor with shape [batch_size x self.state_size].
    """
    raise NotImplementedError("Abstract method")

  @property
  def input_size(self):
    """Integer: size of inputs accepted by this cell."""
    raise NotImplementedError("Abstract method")

  @property
  def output_size(self):
    """Integer: size of outputs produced by this cell."""
    raise NotImplementedError("Abstract method")

  @property
  def state_size(self):
    """Integer: size of state used by this cell."""
    raise NotImplementedError("Abstract method")

  def zero_state(self, batch_size, dtype):
    """Return state tensor (shape [batch_size x state_size]) filled with 0.

    Args:
      batch_size: int, float, or unit Tensor representing the batch size.
      dtype: the data type to use for the state.

    Returns:
      A 2D Tensor of shape [batch_size x state_size] filled with zeros.
    """
    zeros = array_ops.zeros(
        array_ops.pack([batch_size, self.state_size]), dtype=dtype)
    zeros.set_shape([None, self.state_size])
    return zeros


class GRUCell(rnn_cell.RNNCell):
  """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078)."""

  def __init__(self, num_units, input_size=None,
               reset_W_in = init_ops.random_normal_initializer,
               reset_W_hid = init_ops.random_normal_initializer,
               reset_b = init_ops.constant_initializer(0.),
               reset_activation = sigmoid, 
               update_W_in = init_ops.random_normal_initializer,
               update_W_hid = init_ops.random_normal_initializer,
               update_b = init_ops.constant_initializer(0.),
               update_activation = sigmoid
               candidate_W_in = init_ops.random_normal_initializer,
               candidate_W_hid = init_ops.random_normal_initializer,
               candidate_b = init_ops.constant_initializer(0.),
               candidate_activation = tanh):
  self._num_units = num_units
  self._input_size = num_units if input_size is None else input_size
  self._reset_W_in = reset_W_in
  self._reset_W_hid = reset_W_hid
  self._reset_b = reset_b
  self._reset_activation = reset_activation
  self._update_W_in = update_W_in
  self._update_W_hid = update_W_hid
  self._update_b = update_b
  self._update_activation = update_activation
  self._candidate_W_in = candidate_W_in
  self._candidate_W_hid = candidate_W_hid
  self._candidate_b = candidate_b
  self._candidate_activation = candidate_activation

  @property
  def input_size(self):
    return self._input_size

  @property
  def output_size(self):
    return self._num_units

  @property
  def state_size(self):
    return self._num_units

  def _compute(self, args, gate):
    W_in_init, W_hid_init, b_init = gate
    with vs.variable_scope("Reset"):
      W_in = vs.get_variable("W_in",
        [args[0].get_shape()[1], self._num_units],
        initializer=W_in_init)
      W_hid = vs.get_variable("W_hid",
        [args[1].get_shape()[1], self._num_units],
        initializer=W_hid_init)
      b = vs.get_variable("Bias", [self._num_units],
        initializer=b_init)
      matrix = array_ops.concat(0, [W_in, W_hid])
    return matrix, b

  def __call__(self, inputs, state, scope=None):
    """Gated recurrent unit (GRU) with nunits cells."""
    args = [inputs, state]
    with vs.variable_scope(scope or type(self).__name__):  # "GRUCell"
      with vs.variable_scope("Gates"):  # Reset gate and update gate.
        matrices = []
        biases = []
        gates = [
          ("Reset", self._reset_W_in, self._reset_W_hid, self._reset_b)
          ("Update", self._update_W_in, self._update_W_hid, self._update_b)]
        for gate in gates:
          matrix, b = self._compute(args, gate)
          matrices.append(matrix)
          biases.append(bias)
        total_matrix = array_ops.concat(1, matrices)
        total_bias = array_ops.concat(0, biases)
        res_gates = math_ops.matmul(array_ops.concat(1, args), total_matrix)
        res_gates += total_bias
        # We start with bias of 1.0 to not reset and not update.
        r, u = array_ops.split(1, 2, res_gates)
        r, u = self._reset_activation(r), self._update_activation(u)
      with vs.variable_scope("Candidate"):
        candidate = ("Candidate", self._candidate_W_in, self._candidate_W_hid,
            self._candidate_b)
        matrix, b = compute([inputs, r * state], candidate)
        c = math_ops.matmul(array_ops.concat(1, args), matrix) + b
        c = self._candidate_activation(c)
      new_h = u * state + (1 - u) * c
    return new_h, new_h
