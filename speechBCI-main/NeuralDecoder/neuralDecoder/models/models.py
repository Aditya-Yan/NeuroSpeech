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

        # Regularizers
        weightReg = tf.keras.regularizers.L2(weightReg)
        actReg = None  # TF 2.10 does not support actReg on RNNs well

        # Initializers
        recurrent_init = tf.keras.initializers.Orthogonal()
        kernel_init = tf.keras.initializers.GlorotUniform()

        self.subsampleFactor = subsampleFactor
        self.bidirectional = bidirectional
        self.stack_kwargs = stack_kwargs

        # Initial states
        if bidirectional:
            self.initStates = [
                tf.Variable(initial_value=kernel_init(shape=(1, units)), trainable=False),
                tf.Variable(initial_value=kernel_init(shape=(1, units)), trainable=False),
            ]
        else:
            self.initStates = tf.Variable(initial_value=kernel_init(shape=(1, units)), trainable=False)

        # Optional convolutional preprocessing
        self.conv1 = None
        if conv_kwargs is not None:
            self.conv1 = tf.keras.layers.DepthwiseConv1D(
                **conv_kwargs,
                padding='same',
                activation='relu',
                kernel_regularizer=weightReg,
                use_bias=False
            )

        # RNN layers
        self.rnnLayers = []
        for _ in range(nLayers):
            rnn = tf.keras.layers.GRU(
                units,
                return_sequences=True,
                return_state=True,
                kernel_regularizer=weightReg,
                recurrent_initializer=recurrent_init,
                kernel_initializer=kernel_init,
                dropout=dropout
            )
            if bidirectional:
                rnn = tf.keras.layers.Bidirectional(rnn)
            self.rnnLayers.append(rnn)

        # Output layer
        self.dense = tf.keras.layers.Dense(nClasses)

    def call(self, x, states=None, training=False, returnState=False):
        batchSize = tf.shape(x)[0]

        # Optional patch stacking
        if self.stack_kwargs is not None:
            x = tf.image.extract_patches(
                images=x[:, None, :, :],
                sizes=[1, 1, self.stack_kwargs['kernel_size'], 1],
                strides=[1, 1, self.stack_kwargs['strides'], 1],
                rates=[1, 1, 1, 1],
                padding='VALID'
            )
            x = tf.squeeze(x, axis=1)

        # Optional convolution
        if self.conv1 is not None:
            x = self.conv1(x)

        # Handle initial states
        if states is None:
            states = []
            if self.bidirectional:
                states.append([tf.tile(s, [batchSize, 1]) for s in self.initStates])
            else:
                states.append(tf.tile(self.initStates, [batchSize, 1]))
            states.extend([None] * (len(self.rnnLayers) - 1))

        new_states = []
        if self.bidirectional:
            for i, rnn in enumerate(self.rnnLayers):
                outputs = rnn(x, training=training, initial_state=states[i])
                x, forward_s, backward_s = outputs
                if i == len(self.rnnLayers) - 2 and self.subsampleFactor > 1:
                    x = x[:, ::self.subsampleFactor, :]
                new_states.append([forward_s, backward_s])
        else:
            for i, rnn in enumerate(self.rnnLayers):
                outputs = rnn(x, training=training, initial_state=states[i])
                x, s = outputs
                if i == len(self.rnnLayers) - 2 and self.subsampleFactor > 1:
                    x = x[:, ::self.subsampleFactor, :]
                new_states.append(s)

        # Final dense layer
        x = self.dense(x)

        if returnState:
            return x, new_states
        return x

    def getIntermediateLayerOutput(self, x):
        # Example placeholder (fix if needed)
        x, _ = self.rnnLayers[0](x)
        return x

    def getSubsampledTimeSteps(self, timeSteps):
        timeSteps = tf.cast(timeSteps / self.subsampleFactor, dtype=tf.int32)
        if self.stack_kwargs is not None:
            timeSteps = tf.cast(
                (timeSteps - self.stack_kwargs['kernel_size']) / self.stack_kwargs['strides'] + 1,
                dtype=tf.int32
            )
        return timeSteps


class HMRNN(Model):
    """Hierarchical Multi-scale RNN for speech decoding."""

    def __init__(self,
                 units,
                 weightReg,
                 actReg,
                 subsampleFactor,
                 nClasses,
                 bidirectional=False,
                 dropout=0.0,
                 nLayers=3,
                 time_scales=None,
                 conv_kwargs=None,
                 stack_kwargs=None):
        super(HMRNN, self).__init__()

        weightReg = tf.keras.regularizers.L2(weightReg)
        actReg = None

        recurrent_init = tf.keras.initializers.Orthogonal()
        kernel_init = tf.keras.initializers.GlorotUniform()

        self.subsampleFactor = subsampleFactor
        self.bidirectional = bidirectional
        self.stack_kwargs = stack_kwargs
        self.nLayers = nLayers

        # Default time scales
        if time_scales is None:
            time_scales = [1, 4, 16]
        self.time_scales = time_scales[:nLayers]

        # Units per layer
        if isinstance(units, int):
            self.layer_units = [units] * nLayers
        else:
            self.layer_units = units[:nLayers]

        # Optional convolution
        self.conv1 = None
        if conv_kwargs is not None:
            self.conv1 = tf.keras.layers.DepthwiseConv1D(
                **conv_kwargs,
                padding='same',
                activation='relu',
                kernel_regularizer=weightReg,
                use_bias=False
            )

        # Hierarchical RNN layers
        self.hierarchical_rnns = []
        for i in range(nLayers):
            rnn = tf.keras.layers.GRU(
                self.layer_units[i],
                return_sequences=True,
                return_state=True,
                kernel_regularizer=weightReg,
                recurrent_initializer=recurrent_init,
                kernel_initializer=kernel_init,
                dropout=dropout
            )
            if bidirectional:
                rnn = tf.keras.layers.Bidirectional(rnn)
            self.hierarchical_rnns.append(rnn)

        # Output layer
        self.dense = tf.keras.layers.Dense(nClasses)

    def call(self, x, states=None, training=False, returnState=False):
        batchSize = tf.shape(x)[0]

        # Stack patches if needed
        if self.stack_kwargs is not None:
            x = tf.image.extract_patches(
                images=x[:, None, :, :],
                sizes=[1, 1, self.stack_kwargs['kernel_size'], 1],
                strides=[1, 1, self.stack_kwargs['strides'], 1],
                rates=[1, 1, 1, 1],
                padding='VALID'
            )
            x = tf.squeeze(x, axis=1)

        # Optional convolution
        if self.conv1 is not None:
            x = self.conv1(x)

        outputs_at_scales = []
        new_states = []

        for i, (rnn, scale) in enumerate(zip(self.hierarchical_rnns, self.time_scales)):
            # Subsample
            x_subsampled = x[:, ::scale, :] if scale > 1 else x

            # RNN forward
            if self.bidirectional:
                outputs = rnn(x_subsampled, training=training)
                layer_out, forward_s, backward_s = outputs
                new_states.append([forward_s, backward_s])
            else:
                layer_out, layer_state = rnn(x_subsampled, training=training)
                new_states.append(layer_state)

            # Upsample to match base time dimension
            if scale > 1:
                layer_out = tf.repeat(layer_out, repeats=scale, axis=1)
                # Pad or trim to match length
                target_len = tf.shape(x)[1]
                cur_len = tf.shape(layer_out)[1]
                layer_out = layer_out[:, :target_len, :] if cur_len > target_len else tf.concat(
                    [layer_out, tf.zeros([batchSize, target_len - cur_len, tf.shape(layer_out)[2]])],
                    axis=1
                )

            outputs_at_scales.append(layer_out)

        # Combine multi-scale outputs
        combined = tf.concat(outputs_at_scales, axis=-1)

        # Subsample final output
        if self.subsampleFactor > 1:
            combined = combined[:, ::self.subsampleFactor, :]

        # Final output
        output = self.dense(combined)

        if returnState:
            return output, new_states
        return output

    def getSubsampledTimeSteps(self, timeSteps):
        timeSteps = tf.cast(timeSteps / self.subsampleFactor, dtype=tf.int32)
        if self.stack_kwargs is not None:
            timeSteps = tf.cast(
                (timeSteps - self.stack_kwargs['kernel_size']) / self.stack_kwargs['strides'] + 1,
                dtype=tf.int32
            )
        return timeSteps
