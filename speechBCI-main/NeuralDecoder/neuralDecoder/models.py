import tensorflow as tf
from tensorflow.keras import Model


class GRU(Model):
    def __init__(self,
                 units,
                 weightReg,
                 actReg,
                 subsampleFactor,
                 nClasses,
                 bidirectional=False,
                 dropout=0.0,
                 nLayers=2,
                 conv_kwargs=None,
                 stack_kwargs=None):
        super(GRU, self).__init__()

        weightReg = tf.keras.regularizers.L2(weightReg)
        #actReg = tf.keras.regularizers.L2(actReg)
        actReg = None
        recurrent_init = tf.keras.initializers.Orthogonal()
        kernel_init = tf.keras.initializers.glorot_uniform()
        self.subsampleFactor = subsampleFactor
        self.bidirectional = bidirectional
        self.stack_kwargs = stack_kwargs

        if bidirectional:
            self.initStates = [
                kernel_init(shape=(1, units)),  # forward
                kernel_init(shape=(1, units)),  # backward
            ]
        else:
            self.initStates = kernel_init(shape=(1, units))

        self.conv1 = None
        if conv_kwargs is not None:
            self.conv1 = tf.keras.layers.DepthwiseConv1D(
                                                **conv_kwargs,
                                               padding='same',
                                               activation='relu',
                                               kernel_regularizer=weightReg,
                                               use_bias=False)

        self.rnnLayers = []
        for _ in range(nLayers):
            # create cell (cell-level args only)
            cell = tf.keras.layers.GRUCell(
                units,
                kernel_regularizer=weightReg,
                recurrent_initializer=recurrent_init,
                kernel_initializer=kernel_init,
                # do NOT pass return_sequences/return_state here
            )

            # wrap cell in RNN layer which exposes return_sequences & return_state
            rnn = tf.keras.layers.RNN(cell, return_sequences=True, return_state=True)

            # if bidirectional, wrap the RNN layer (this returns output, fw_state, bw_state)
            if bidirectional:
                rnn = tf.keras.layers.Bidirectional(rnn, merge_mode='concat')

            self.rnnLayers.append(rnn)
        self.dense = tf.keras.layers.Dense(nClasses)

    def call(self, x, states=None, training=False, returnState=False):
        batchSize = tf.shape(x)[0]

        if self.stack_kwargs is not None:
            x = tf.image.extract_patches(x[:, None, :, :],
                                         sizes=[1, 1, self.stack_kwargs['kernel_size'], 1],
                                         strides=[1, 1, self.stack_kwargs['strides'], 1],
                                         rates=[1, 1, 1, 1],
                                         padding='VALID')
            x = tf.squeeze(x, axis=1)

        if self.conv1 is not None:
            x = self.conv1(x)

        if states is None:
            states = []
            for layer_idx in range(len(self.rnnLayers)):
                if self.bidirectional:
                    # append forward state then backward state for this layer
                    states.append(tf.tile(self.initStates[0], [batchSize, 1]))  # fw
                    states.append(tf.tile(self.initStates[1], [batchSize, 1]))  # bw
                else:
                    # append single initial state for this layer
                    states.append(tf.tile(self.initStates, [batchSize, 1]))


        new_states = []
        if self.bidirectional:
            for layer_idx, rnn in enumerate(self.rnnLayers):
                # Grab forward & backward state for this layer
                fw_state = states[2*layer_idx]
                bw_state = states[2*layer_idx + 1]
                x, forward_s, backward_s = rnn(x, training=training, initial_state=[fw_state, bw_state])
                new_states.extend([forward_s, backward_s])

                if layer_idx == len(self.rnnLayers) - 2 and self.subsampleFactor > 1:
                    x = x[:, ::self.subsampleFactor, :]
        else:
            for layer_idx, rnn in enumerate(self.rnnLayers):
                st = states[layer_idx]
                x, s = rnn(x, training=training, initial_state=[st])
                if layer_idx == len(self.rnnLayers) - 2 and self.subsampleFactor > 1:
                    x = x[:, ::self.subsampleFactor, :]
                new_states.append(s)

        x = self.dense(x, training=training)

        if returnState:
            return x, new_states
        else:
            return x

    # TODO: Fix me
    def getIntermediateLayerOutput(self, x):
        x, _ = self.rnn1(x)
        return x

    def getSubsampledTimeSteps(self, timeSteps):
        timeSteps = tf.cast(timeSteps / self.subsampleFactor, dtype=tf.int32)
        if self.stack_kwargs is not None:
            timeSteps = tf.cast((timeSteps - self.stack_kwargs['kernel_size']) / self.stack_kwargs['strides'] + 1, dtype=tf.int32)
        return timeSteps