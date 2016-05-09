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


class Gate(object):
  """Gate to handle to handle initialization"""  

  def __init__(self, W_in=init_ops.random_normal_initializer(stddev=0.1),
               W_hid=init_ops.random_normal_initializer(stddev=0.1),
               W_cell=init_ops.random_normal_initializer(stddev=0.1),
               b=init_ops.constant_initializer(0.),
               activation=None):
    self.W_in = W_in
    self.W_hid = W_hid
    # Don't store a cell weight vector when cell is None
    if W_cell is not None:
        self.W_cell = W_cell
    if b is not None:
      self.b = b
    # For the activation, if None is supplied, use identity
    if activation is None:
        self.activation = control_flow_ops.identity
    else:
        self.activation = activation

class GRUCell(rnn_cell.RNNCell):
  """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078)."""

  def __init__(self, num_units, input_size=None,
               resetgate=Gate(W_cell=None, activation=sigmoid),
               updategate=Gate(W_cell=None, activation=sigmoid),
               candidategate=Gate(W_cell=None, activation=tanh)):
    self._num_units = num_units
    self._input_size = num_units if input_size is None else input_size
    self._resetgate = resetgate
    self._updategate = updategate
    self._candidategate = candidategate

  @property
  def input_size(self):
    return self._input_size

  @property
  def output_size(self):
    return self._num_units

  @property
  def state_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    """Gated recurrent unit (GRU) with nunits cells."""
    with vs.variable_scope(scope or type(self).__name__):  # "GRUCell"
      with vs.variable_scope("Gates"):  # Reset gate and update gate.
        # We start with bias of 1.0 to not reset and not update.
        r, u = array_ops.split(1, 2, Modified_linear([inputs, state],
          [(self._num_units, "Reset", self._resetgate),
           (self._num_units, "Update", self._updategate)]))
        r, u = self._resetgate.activation(r), self._updategate.activation(u)
      with vs.variable_scope("Candidate"):
        c = Modified_linear([inputs, r * state],
          (self._num_units, "Candidate", self._candidategate))
        c = self._candidategate.activation(c)
      new_h = u * state + (1 - u) * c
    return new_h, new_h


def Modified_linear(args, output, scope=None):
  """Modified linear takes args and output.
     Args is same as in linear, but output is a tuple consisting of:
     output_size, name of gate, gate object (with all initializations)
  """
  if args is None or (isinstance(args, (list, tuple)) and not args):
    raise ValueError("`args` must be specified")
  if not isinstance(args, (list, tuple)):
    args = [args]
  if not isinstance(output, list):
    output = [output]
  shapes = [a.get_shape().as_list() for a in args]
  for shape in shapes:
    if len(shape) != 2:
      raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
    if not shape[1]:
      raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))

  matrices = []
  biases = []
  with vs.variable_scope(scope or "Linear"):
    for output_size, name, gate in output: # loops over every gate
      with vs.variable_scope(name):
        W_in = vs.get_variable("W_in", [args[0].get_shape()[1], output_size],
          initializer=gate.W_in)
        W_hid = vs.get_variable("W_hid", [args[1].get_shape()[1], output_size],
          initializer=gate.W_hid)
        if hasattr(gate, 'b'):
          b = vs.get_variable("Bias", [output_size],
            initializer=gate.b)
          biases.append(b)
        if hasattr(gate, "W_cell"):
          pass
          # do some LSTM stuff ...
        else:
          matrix = array_ops.concat(0, [W_in, W_hid]) # concats all matrices
        matrices.append(matrix)

  total_matrix = array_ops.concat(1, matrices) # concats across gates
  res = math_ops.matmul(array_ops.concat(1, args), total_matrix) # computes the results

  if biases is not []:
    total_bias = array_ops.concat(0, biases) # concats across gates biases
    if total_matrix.get_shape()[1] != total_bias.get_shape()[0]:
      raise ValueError('Must have same output dimensions for W and b')
    res += total_bias
  return res