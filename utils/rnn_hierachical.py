import tensorflow as tf
from tensorflow.python.ops import tensor_array_ops
from utils.tfextensions import mask

def attn(p1, inp_state, W_a, b_a, v_a, h_lengths, hidden):
    p2 = tf.matmul(inp_state, W_a) + b_a
    p2 = tf.expand_dims(p2, 1)
    john = p1 + p2
    e = tf.reduce_sum(v_a * tf.tanh(john), [2])
    alpha = tf.nn.softmax(e)
    alpha = tf.to_float(mask(h_lengths)) * alpha
    alpha = alpha / tf.reduce_sum(alpha, [1], keep_dims=True)
    c = tf.reduce_sum(tf.expand_dims(alpha, 2) * tf.squeeze(hidden), [1])
    return c, alpha

def rnn_step(old_step, inp, W_hh, W_xh, b_h):
    next_step = tf.tanh(tf.matmul(old_step, W_hh) + tf.matmul(inp, W_xh) + b_h)
    return next_step

def gru_step(old_step, inp, W_z, W_r, W_h, b_z, b_r, b_h):
    con = tf.concat(1, [inp, old_step])
    z = tf.sigmoid(tf.matmul(con, W_z) + b_z)
    r = tf.sigmoid(tf.matmul(con, W_r) + b_r)
    con = tf.concat(1, [inp, r*old_step])
    h = tf.tanh(tf.matmul(con, W_h) + b_h)
    new_step = (1-z)*h + z*old_step
    return new_step, z, r

def attention_decoder(h_input, h_lengths, h_state, num_h, target_input,
                      target_input_lengths, num_attn_units, num_attn_rnn_units,
                      embeddings, W_out, b_out, name='decoder',
                      same_attention_weights=False, swap=False):
    """Decoder with attention.

    Note that the number of units in the attention decoder must always
    be equal to the size of the initial state/attention input.

    Keyword arguments:
        attention_input:    the input to put attention on. expected dims: [batch_size, attention_length, attention_dims]
        initial_state:      The initial state for the decoder RNN.
        target_input:       The target to replicate. Expected: [batch_size, max_target_sequence_len, embedding_dims]
        num_attn_units:     Number of units in the alignment layer that produces the context vectors.
    """
    with tf.variable_scope(name):
        #h0_input = attention_inputs[0]
        #h0_lengths = attention_lengths[0]
        #h0_state = initial_state[0]
        #h1_input = attention_inputs[1]
        #h1_lengths = attention_lengths[1]
        #h1_state = initial_state[1]
        target_dims = target_input.get_shape()[2]
        max_sequence_length = tf.reduce_max(target_input_lengths)
        print(h_lengths)
        h_dims = dict()
        num_units = dict()
        h_len = dict()

        for i in range(num_h):
            h_dims[i] = h_input[i].get_shape()[2]
            num_units[i] = h_dims[i]
            h_len[i] = tf.shape(h_input[i])[1]

        weight_initializer = tf.truncated_normal_initializer(stddev=0.1)

        W_dz = tf.get_variable('W_dz',
                              shape=[target_dims+num_units[0]+num_attn_rnn_units, num_units[0]],
                              initializer=weight_initializer)
        W_dr = tf.get_variable('W_dr',
                              shape=[target_dims+num_units[0]+num_attn_rnn_units, num_units[0]],
                              initializer=weight_initializer)
        W_dh = tf.get_variable('W_dh',
                              shape=[target_dims+num_units[0]+num_attn_rnn_units, num_units[0]],
                              initializer=weight_initializer)
        b_dz = tf.get_variable('b_dz',
                              shape=[num_units[0]],
                              initializer=tf.constant_initializer(1.0))
        b_dr = tf.get_variable('b_dr',
                              shape=[num_units[0]],
                              initializer=tf.constant_initializer(1.0))
        b_dh = tf.get_variable('b_dh',
                              shape=[num_units[0]],
                              initializer=tf.constant_initializer())

        # for attention of hierachies part
        W_a = dict()
        U_a = dict()
        b_a = dict()
        v_a = dict()

        W_a[0] = tf.get_variable('W_a0',
                                   shape=[num_attn_rnn_units, num_attn_units],
                                   initializer=weight_initializer)
        U_a[0] = tf.get_variable('U_a0',
                                   shape=[1, 1, h_dims[0], num_attn_units],
                                   initializer=weight_initializer)
        b_a[0] = tf.get_variable('b_a0',
                                   shape=[num_attn_units],
                                   initializer=tf.constant_initializer())
        v_a[0] = tf.get_variable('v_a0',
                                   shape=[num_attn_units],
                                   initializer=weight_initializer)

        # h0 attention

        for i in range(1, num_h):
            if same_attention_weights:
                W_a[i] = W_a[0] # Just reusing the same weights
                U_a[i] = U_a[0]
                b_a[i] = b_a[0]
                v_a[i] = v_a[0]
            else:
                W_a[i] = tf.get_variable(('W_a%d' % i),
                                         shape=[num_attn_rnn_units, num_attn_units],
                                         initializer=weight_initializer)
                U_a[i] = tf.get_variable(('U_a%d' % i),
                                         shape=[1, 1, h_dims[i], num_attn_units],
                                         initializer=weight_initializer)
                b_a[i] = tf.get_variable(('b_a%d' % i),
                                         shape=[num_attn_units],
                                         initializer=tf.constant_initializer())
                v_a[i] = tf.get_variable(('v_a%d' % i),
                                         shape=[num_attn_units],
                                         initializer=weight_initializer)

        # for attention of recurrent part
        # to make state fit
        W_as = tf.get_variable('W_as',
                               shape=[h_dims[0], num_attn_rnn_units],
                               initializer=weight_initializer)
        b_as = tf.get_variable('b_as',
                               shape=[num_attn_rnn_units],
                               initializer=tf.constant_initializer())
        # hidden weights
        W_ahh = tf.get_variable('W_ahh',
                                shape=[num_attn_rnn_units, num_attn_rnn_units],
                                initializer=weight_initializer)
        W_axh = tf.get_variable('W_axh',
                                shape=[h_dims[0], num_attn_rnn_units],
                                initializer=weight_initializer)
        b_ah = tf.get_variable('b_ah',
                               shape=[num_attn_rnn_units],
                               initializer=tf.constant_initializer())

        W_az = tf.get_variable('W_az',
                              shape=[h_dims[0]+num_attn_rnn_units, num_attn_rnn_units],
                              initializer=weight_initializer)
        W_ar = tf.get_variable('W_ar',
                              shape=[h_dims[0]+num_attn_rnn_units, num_attn_rnn_units],
                              initializer=weight_initializer)
        W_ah = tf.get_variable('W_ah',
                              shape=[h_dims[0]+num_attn_rnn_units, num_attn_rnn_units],
                              initializer=weight_initializer)
        b_az = tf.get_variable('b_az',
                              shape=[num_attn_rnn_units],
                              initializer=tf.constant_initializer(1.0))
        b_ar = tf.get_variable('b_ar',
                              shape=[num_attn_rnn_units],
                              initializer=tf.constant_initializer(1.0))
        #b_ah = tf.get_variable('b_ah',
        #                      shape=[num_attn_rnn_units],
        #                      initializer=tf.constant_initializer())

        # TODO: don't use convolutions!
        # TODO: fix the bias (b_a)
        hidden = dict()
        part1 = dict()
        for i in range(num_h):
            hidden[i] = tf.reshape(h_input[i], tf.pack([-1, h_len[i], 1, h_dims[i]]))
            part1[i] = tf.nn.conv2d(hidden[i], U_a[i], [1, 1, 1, 1], "SAME")
            part1[i] = tf.squeeze(part1[i], [2]) # squeeze out the third dimension

        inputs = tf.transpose(target_input, perm=[1, 0, 2])
        input_ta = tensor_array_ops.TensorArray(tf.float32, size=1, dynamic_size=True)
        input_ta = input_ta.unpack(inputs)

        def decoder_cond(time, state, output_ta_t, *args):#, a_tracker, az_tracker, ar_tracker):
            return tf.less(time, max_sequence_length)

        def decoder_body_builder(feedback=False):
            def decoder_body(time, old_state, output_ta_t, *args):#, a_tracker, az_tracker, ar_tracker):
                if feedback:
                    def from_previous():
                        prev_1 = tf.matmul(old_state, W_out) + b_out
                        return tf.gather(embeddings, tf.argmax(prev_1, 1))
                    x_t = tf.cond(tf.greater(time, 0), from_previous, lambda: input_ta.read(0))
                else:
                    x_t = input_ta.read(time)
                args = list(args)
                # attention
                start_rnn = tf.tanh(tf.matmul(old_state, W_as) + b_as)
                prev_hidden = start_rnn
                for i in reversed(range(num_h)):
                    ac, alpha = attn(p1=part1[i], inp_state=prev_hidden, W_a=W_a[i], b_a=b_a[i], v_a=v_a[i], h_lengths=h_lengths[i], hidden=hidden[i])
                    prev_hidden, az, ar = gru_step(prev_hidden, ac, W_az, W_ar, W_ah, b_az, b_ar, b_ah)
                    args[i*3] = args[i*3].write(time, alpha)
                    args[i*3+1] = args[i*3+1].write(time, az)
                    args[i*3+2] = args[i*3+2].write(time, ar)
                    #az_tracker[i] = az_tracker[i].write(time, az)
                    #ar_tracker[i] = ar_tracker[i].write(time, ar)

                c = prev_hidden # will get last hidden out

                # GRU
                con = tf.concat(1, [x_t, old_state, c])
                z = tf.sigmoid(tf.matmul(con, W_dz) + b_dz)
                r = tf.sigmoid(tf.matmul(con, W_dr) + b_dr)
                con = tf.concat(1, [x_t, r*old_state, c])
                h = tf.tanh(tf.matmul(con, W_dh) + b_dh)
                new_state = (1-z)*h + z*old_state

                output_ta_t = output_ta_t.write(time, new_state)

                return (time + 1, new_state, output_ta_t) + tuple(args)#, a_tracker, az_tracker, ar_tracker)
            return decoder_body


        output_ta = tensor_array_ops.TensorArray(tf.float32, size=1, dynamic_size=True)
        if num_h == 1:
            a_list = [1, 1, 1]
        if num_h == 5:
            a_list = [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5]

        for i in range(num_h):
            a_list[i*3+0] = tensor_array_ops.TensorArray(tf.float32, size=1, dynamic_size=True)
            a_list[i*3+1] = tensor_array_ops.TensorArray(tf.float32, size=1, dynamic_size=True)
            a_list[i*3+2] = tensor_array_ops.TensorArray(tf.float32, size=1, dynamic_size=True)

        time = tf.constant(0)
        loop_vars = [time, h_state[0], output_ta] + a_list

        _, state, output_ta, *_ = (#, _, _, _ = 
            tf.while_loop(decoder_cond,
                          decoder_body_builder(),
                          loop_vars,
                          swap_memory=swap))
        _, valid_state, valid_output_ta, *valid_a_data = (#, valid_a_tracker, valid_az_tracker, valid_ar_tracker 
            tf.while_loop(decoder_cond,
                decoder_body_builder(feedback=True),
                loop_vars,
                swap_memory=swap))

        dec_state = state
        dec_out = tf.transpose(output_ta.pack(), perm=[1, 0, 2])
        valid_dec_out = tf.transpose(valid_output_ta.pack(), perm=[1, 0, 2])

        #valid_a_tracker = dict()
        #valid_az_tracker = dict()
        #valid_ar_tracker = dict()

        #for i in range(num_h):
        #    valid_a_tracker[i] = valid_a_data[i*3].pack()
        #    valid_az_tracker[i] = valid_a_data[i*3+1].pack()
        #    valid_ar_tracker[i] = valid_a_data[i*3+2].pack()

        return dec_state, dec_out, valid_dec_out, valid_a_data#valid_a_tracker, valid_ar_tracker, valid_az_tracker
