#Aidan Fike
#June 12, 2019

#Functions to create layers used in the convolutional neural network
#These were adapted from @author: hlt-titan

import tensorflow as tf
from tensorflow.python.util import nest

#A function to create a signal cnn layer using a leaky relu activation function
#
#Params: X - the input to the network
#        img_h - the number of amino acids used to represent the sequence
#        img_w - the number of different types of amino acids
#        filter_hs - the heights of the various filters
#        batch_size - the number of instances being passed in 
#        num_filter - the number of each type of filter that is applied
#        leakSlope - the slope of the leaky relu
#
#Return - The created layer
def multiChannelCnn(X, img_h, img_w, filter_hs, batch_size, num_filter, leakSlope):
    outputs = []
    cnn_input = tf.reshape(X, [-1, img_h, img_w, 1])

    #For each filter height, first pad the instances with a number of zeros
    #such that the same height will be outputted as that which is inputted
    for i, filter_h in enumerate(filter_hs):
        with tf.variable_scope("cnn_channel{0}".format(i)):
            #Pad the inputs
            up_pad = tf.zeros([batch_size, int(filter_hs[i]/2), img_w, 1])
            if filter_hs[i]%2==1:
                down_pad = tf.zeros([batch_size, int(filter_hs[i]/2), img_w, 1])
            else:
                down_pad = tf.zeros([batch_size, int(filter_hs[i]/2)-1, img_w, 1])
            L_input = tf.concat(axis=1, values=[up_pad, cnn_input, down_pad])

            #Initialize the weights then apply them
            init = tf.truncated_normal_initializer(stddev=2.0/(img_h*img_w))
            W_conv = tf.get_variable(
                "W_conv", [filter_hs[i], img_w, 1, num_filter],initializer=init)
            b_conv = tf.get_variable("b_conv", [num_filter], initializer=tf.constant_initializer(0.0))
            L_scores = tf.nn.conv2d(L_input, W_conv, strides = [1,1,1,1], padding = "VALID") + b_conv
            L_relu = tf.nn.leaky_relu(L_scores, alpha=leakSlope) #Leaky relu activation
            L_out = tf.squeeze(L_relu, [2])
            outputs.append(L_out)
    out = tf.concat(axis=2, values=outputs)
    return out

#Create a simple fully connected layer 
#
#Params: args: a 2D Tensor or a list of 2D, batch x n, Tensors.
#        output_size: int, second dimension of W[i].
#        bias: boolean, whether to add a bias term or not.
#        bias_start: starting value to initialize the bias; 0 by default.
#        scope: VariableScope for the created subgraph; defaults to "Linear".
#
#Returns:
#    A 2D Tensor with shape [batch x output_size] equal to
#    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
#
#Raises:
#    ValueError: if some of the arguments has unspecified or wrong shape.
def linear(args, output_size, bias, bias_start=0.0, scope=None):
  if args is None or (nest.is_sequence(args) and not args):
    raise ValueError("`args` must be specified")
  if not nest.is_sequence(args):
    args = [args]

  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape().as_list() for a in args]
  for shape in shapes:
    if len(shape) != 2:
      raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
    if not shape[1]:
      raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
    else:
      total_arg_size += shape[1]

  dtype = [a.dtype for a in args][0]

  # Now the computation.
  with tf.variable_scope(scope or "Linear"):
    matrix = tf.get_variable(
        "Matrix", [total_arg_size, output_size], dtype=dtype)
    if len(args) == 1:
      res = tf.matmul(args[0], matrix)
    else:
      res = tf.matmul(tf.concat(1, args), matrix)
    if not bias:
      return res
    bias_term = tf.get_variable(
        "Bias", [output_size],
        dtype=dtype,
        initializer=tf.constant_initializer(
            bias_start, dtype=dtype))
  return res + bias_term
