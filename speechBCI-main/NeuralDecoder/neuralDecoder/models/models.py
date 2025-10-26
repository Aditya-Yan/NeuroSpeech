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
                tf.Variable(initial_value=kernel_init(shape=(1, units))),
                tf.Variable(initial_value=kernel_init(shape=(1, units))),
            ]
        else:
            self.initStates = tf.Variable(initial_value=kernel_init(shape=(1, units)))

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
            rnn = tf.keras.layers.GRU(units,
                                      return_sequences=True,
                                      return_state=True,
                                      kernel_regularizer=weightReg,
                                      activity_regularizer=actReg,
                                      recurrent_initializer=recurrent_init,
                                      kernel_initializer=kernel_init,
                                      dropout=dropout)
            self.rnnLayers.append(rnn)
        if bidirectional:
            self.rnnLayers = [tf.keras.layers.Bidirectional(rnn) for rnn in self.rnnLayers]
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
            if self.bidirectional:
                states.append([tf.tile(s, [batchSize, 1]) for s in self.initStates])
            else:
                states.append(tf.tile(self.initStates, [batchSize, 1]))
            states.extend([None] * (len(self.rnnLayers) - 1))

        new_states = []
        if self.bidirectional:
            for i, rnn in enumerate(self.rnnLayers):
                x, forward_s, backward_s = rnn(x, training=training, initial_state=states[i])
                if i == len(self.rnnLayers) - 2:
                    if self.subsampleFactor > 1:
                        x = x[:, ::self.subsampleFactor, :]
                new_states.append([forward_s, backward_s])
        else:
            for i, rnn in enumerate(self.rnnLayers):
                x, s = rnn(x, training=training, initial_state=states[i])
                if i == len(self.rnnLayers) - 2:
                    if self.subsampleFactor > 1:
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


class HMRNN(Model):
    """
    Hierarchical Multi-scale RNN for speech decoding.
    Processes input at multiple time scales (fast, medium, slow).
    """
    
    def __init__(self,
                 units,                # Can be int or list [fast, medium, slow]
                 weightReg,
                 actReg,
                 subsampleFactor,
                 nClasses,
                 bidirectional=False,
                 dropout=0.0,
                 nLayers=3,            # Number of hierarchical layers
                 time_scales=None,     # [1, 4, 16] - subsampling for each layer
                 conv_kwargs=None,
                 stack_kwargs=None):
        super(HMRNN, self).__init__()

        weightReg = tf.keras.regularizers.L2(weightReg)
        actReg = None
        recurrent_init = tf.keras.initializers.Orthogonal()
        kernel_init = tf.keras.initializers.glorot_uniform()
        
        self.subsampleFactor = subsampleFactor
        self.bidirectional = bidirectional
        self.stack_kwargs = stack_kwargs
        self.nLayers = nLayers
        
        # Default time scales: fast (every step), medium (every 4), slow (every 16)
        if time_scales is None:
            time_scales = [1, 4, 16]
        self.time_scales = time_scales[:nLayers]  # Take only nLayers scales
        
        # Handle units - can be single int or list
        if isinstance(units, int):
            self.layer_units = [units] * nLayers
        else:
            self.layer_units = units[:nLayers]
        
        # Input convolution (optional)
        self.conv1 = None
        if conv_kwargs is not None:
            self.conv1 = tf.keras.layers.DepthwiseConv1D(
                **conv_kwargs,
                padding='same',
                activation='relu',
                kernel_regularizer=weightReg,
                use_bias=False)
        
        # Create hierarchical RNN layers
        self.hierarchical_rnns = []
        for i in range(nLayers):
            rnn = tf.keras.layers.GRU(
                self.layer_units[i],
                return_sequences=True,
                return_state=True,
                kernel_regularizer=weightReg,
                activity_regularizer=actReg,
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
        """
        Forward pass through hierarchical layers.
        Each layer processes input at different time scales.
        """
        batchSize = tf.shape(x)[0]
        original_length = tf.shape(x)[1]
        
        # Stack/patch input if specified
        if self.stack_kwargs is not None:
            x = tf.image.extract_patches(
                x[:, None, :, :],
                sizes=[1, 1, self.stack_kwargs['kernel_size'], 1],
                strides=[1, 1, self.stack_kwargs['strides'], 1],
                rates=[1, 1, 1, 1],
                padding='VALID'
            )
            x = tf.squeeze(x, axis=1)
        
        # Input convolution
        if self.conv1 is not None:
            x = self.conv1(x)
        
        # Store outputs from each scale
        outputs_at_scales = []
        new_states = []
        
        # Process through each hierarchical layer
        for i, (rnn, scale) in enumerate(zip(self.hierarchical_rnns, self.time_scales)):
            # Subsample input for higher layers (coarser time scales)
            if scale > 1:
                x_subsampled = x[:, ::scale, :]
            else:
                x_subsampled = x
            
            # Process through RNN at this time scale
            if self.bidirectional:
                layer_out, forward_s, backward_s = rnn(x_subsampled, training=training)
                new_states.append([forward_s, backward_s])
            else:
                layer_out, layer_state = rnn(x_subsampled, training=training)
                new_states.append(layer_state)
            
            # Upsample back to original time resolution
            if scale > 1:
                # Repeat each time step 'scale' times
                layer_out = tf.repeat(layer_out, repeats=scale, axis=1)
                
                # Trim or pad to match original length
                current_length = tf.shape(layer_out)[1]
                if current_length > tf.shape(x)[1]:
                    layer_out = layer_out[:, :tf.shape(x)[1], :]
                elif current_length < tf.shape(x)[1]:
                    padding = tf.zeros([batchSize, tf.shape(x)[1] - current_length, tf.shape(layer_out)[2]])
                    layer_out = tf.concat([layer_out, padding], axis=1)
            
            outputs_at_scales.append(layer_out)
        
        # Combine outputs from all time scales
        # Concatenate along feature dimension
        combined = tf.concat(outputs_at_scales, axis=-1)
        
        # Apply final subsampling if needed
        if self.subsampleFactor > 1:
            combined = combined[:, ::self.subsampleFactor, :]
        
        # Final output layer
        output = self.dense(combined, training=training)
        
        if returnState:
            return output, new_states
        else:
            return output

    def getSubsampledTimeSteps(self, timeSteps):
        """Calculate output time steps after subsampling"""
        timeSteps = tf.cast(timeSteps / self.subsampleFactor, dtype=tf.int32)
        if self.stack_kwargs is not None:
            timeSteps = tf.cast(
                (timeSteps - self.stack_kwargs['kernel_size']) / self.stack_kwargs['strides'] + 1,
                dtype=tf.int32
            )
        return timeSteps
  
