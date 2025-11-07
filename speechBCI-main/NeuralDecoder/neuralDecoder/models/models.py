# models.py
import tensorflow as tf
from tensorflow.keras import Model
from typing import List, Optional

# -------------------------
# Helper: Straight-through rounding for discrete z
# -------------------------
def round_ste(x):
    """
    Straight-through estimator for rounding: returns discrete 0/1 values
    but allows gradient of the original sigmoid (x) to flow back.
    """
    x_round = tf.round(x)
    return x_round + tf.stop_gradient(x - x_round)

# -------------------------
# Standard GRU model (kept for compatibility)
# -------------------------
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
        actReg = None  # TF 2.10 compatibility

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
                # outputs -> (sequence, forward_state, backward_state)
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

# -------------------------
# HM-GRU layer with learned boundary detector
# -------------------------
class HMGRULayer(tf.keras.layers.Layer):
    """
    Hierarchical GRU layer with a learned boundary detector per timestep.
    - Uses a GRUCell to compute candidate new state.
    - Computes a boundary probability z_prob = sigmoid(W_z [x_t; h_{t-1}])
    - Discretizes z via rounding with STE (round_ste)
    - Updates the state: h_t = z * h_new + (1 - z) * h_prev
    """
    def __init__(self, units, weightReg=0.0, dropout=0.0, name=None):
        super(HMGRULayer, self).__init__(name=name)
        self.units = units
        self.weightReg = tf.keras.regularizers.L2(weightReg) if weightReg is not None and weightReg > 0 else None
        self.dropout = dropout

        # GRU cell for candidate updates
        self.gru_cell = tf.keras.layers.GRUCell(units,
                                                kernel_regularizer=self.weightReg,
                                                recurrent_initializer=tf.keras.initializers.Orthogonal(),
                                                kernel_initializer=tf.keras.initializers.GlorotUniform())
        # Boundary detector: gives scalar logit per batch element
        self.boundary_net = tf.keras.layers.Dense(1,
                                                  activation=None,
                                                  kernel_initializer=tf.keras.initializers.GlorotUniform(),
                                                  kernel_regularizer=self.weightReg)

    def build(self, input_shape):
        # input_shape: (batch, features) for a single timestep
        super(HMGRULayer, self).build(input_shape)

    def call_step(self, x_t, h_prev):
        """
        Compute one timestep update for the layer.
        x_t: [batch, feat]
        h_prev: [batch, units]
        returns: h_t, z_hard, z_prob
        """
        # Candidate update via GRUCell (returns new candidate state)
        _, h_candidate = self.gru_cell(x_t, [h_prev])

        # Boundary probability: use concatenation of input and prev hidden
        concat = tf.concat([x_t, h_prev], axis=-1)
        z_logit = self.boundary_net(concat)  # [batch, 1]
        z_prob = tf.sigmoid(z_logit)
        z_hard = round_ste(z_prob)  # discrete 0/1 with STE

        # Broadcast z_hard to match hidden size
        z_b = tf.cast(z_hard, dtype=h_candidate.dtype)  # [batch,1]
        h_t = z_b * h_candidate + (1.0 - z_b) * h_prev
        h_t = tf.cast(h_t, dtype=h_candidate.dtype)
        return h_t, tf.squeeze(z_hard, axis=-1), tf.squeeze(z_prob, axis=-1)

# -------------------------
# Hierarchical Multi-scale RNN with learned boundaries
# -------------------------
class HMRNN(Model):
    """
    HMRNN: stack of HMGRULayers that implement hierarchical boundary-driven updates.
    Each lower layer produces boundary signals z^l_t; higher layer updates only when z^{l-1}_t == 1.
    The final output is the concatenation of each layer's hidden state at every timestep, projected by a Dense layer.
    """

    def __init__(self,
                 units,
                 weightReg,
                 actReg,
                 subsampleFactor,
                 nClasses,
                 bidirectional=False,
                 dropout=0.0,
                 nLayers=3,
                 time_scales: Optional[List[int]] = None,
                 conv_kwargs=None,
                 stack_kwargs=None):
        super(HMRNN, self).__init__()

        # Regularizers/initializers
        self.weightReg = weightReg
        self.subsampleFactor = subsampleFactor
        self.bidirectional = bidirectional  # Not used for HM-RNN currently
        self.stack_kwargs = stack_kwargs
        self.nLayers = nLayers

        # Time scales are now informational only — actual dynamic boundaries drive updates.
        if time_scales is None:
            time_scales = [1] * nLayers
        self.time_scales = time_scales[:nLayers]

        # Units per layer
        if isinstance(units, int):
            self.layer_units = [units] * nLayers
        else:
            self.layer_units = units[:nLayers]

        # Optional conv preprocessing
        self.conv1 = None
        if conv_kwargs is not None:
            self.conv1 = tf.keras.layers.DepthwiseConv1D(
                **conv_kwargs,
                padding='same',
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.L2(weightReg) if weightReg else None,
                use_bias=False
            )

        # Build hierarchical layers
        self.h_layers = []
        for i in range(nLayers):
            layer = HMGRULayer(self.layer_units[i], weightReg=weightReg, dropout=dropout, name=f"hmgrulayer_{i}")
            self.h_layers.append(layer)

        # Final projection
        # We'll project the concatenated hidden states from all layers to nClasses
        total_units = sum(self.layer_units)
        self.dense = tf.keras.layers.Dense(nClasses, kernel_initializer=tf.keras.initializers.GlorotUniform())

    def call(self, x, states=None, training=False, returnState=False):
        """
        x: [batch, time, features]
        states: optional list of initial hidden states for each layer (list of tensors [batch, units])
        """
        batch_size = tf.shape(x)[0]
        time_len = tf.shape(x)[1]

        # Patch stacking if requested
        if self.stack_kwargs is not None:
            x = tf.image.extract_patches(
                images=x[:, None, :, :],
                sizes=[1, 1, self.stack_kwargs['kernel_size'], 1],
                strides=[1, 1, self.stack_kwargs['strides'], 1],
                rates=[1, 1, 1, 1],
                padding='VALID'
            )
            x = tf.squeeze(x, axis=1)

        # Optional conv preprocessing
        if self.conv1 is not None:
            x = self.conv1(x)

        # Initialize states if not provided
        if states is None:
            states = []
            for u in self.layer_units:
                # initialize with zeros
                states.append(tf.zeros([batch_size, u], dtype=tf.float32))

        # Prepare outputs storage
        ta_outputs = [tf.TensorArray(dtype=tf.float32, size=time_len) for _ in range(self.nLayers)]
        ta_z = [tf.TensorArray(dtype=tf.int32, size=time_len) for _ in range(self.nLayers)]
        ta_z_prob = [tf.TensorArray(dtype=tf.float32, size=time_len) for _ in range(self.nLayers)]

        # Unstack inputs along time
        x_time = tf.unstack(x, axis=1)  # list length T of [batch, features]

        # Iterate over time steps
        # At each time step:
        #  - Layer 0 always receives x_t and updates (computes z0)
        #  - Layer l > 0 updates only where z_{l-1} == 1; its input is h_{l-1,t}
        h_states = list(states)  # current states per layer
        for t_idx in tf.range(time_len):
            x_t = x_time[t_idx]  # [batch, features]

            # Layer 0 update (always)
            h0_prev = h_states[0]
            h0_new, z0_hard, z0_prob = self.h_layers[0].call_step(x_t, h0_prev)
            h_states[0] = h0_new
            ta_outputs[0] = ta_outputs[0].write(t_idx, h0_new)
            ta_z[0] = ta_z[0].write(t_idx, tf.cast(z0_hard, tf.int32))
            ta_z_prob[0] = ta_z_prob[0].write(t_idx, z0_prob)

            # Higher layers
            for L in range(1, self.nLayers):
                # If lower-layer z==1 -> update this layer with lower-layer hidden representation
                z_lower = ta_z[L-1].read(t_idx)  # scalar per batch? it's Tensor shape [batch], but read returns tf.Tensor
                # read returns shape [batch] (int)
                # We create mask of which batch elements should update
                z_mask = tf.cast(z_lower, tf.float32)[:, None]  # [batch, 1]

                # Input to this layer (use lower-layer hidden state)
                input_L = h_states[L-1]

                h_prev = h_states[L]
                # compute candidate and z for this layer using input_L and its previous state
                h_candidate, z_hard_L, z_prob_L = self.h_layers[L].call_step(input_L, h_prev)

                # Only apply updates where z_lower == 1
                # If z_lower == 1 -> layer L uses its own z_hard_L to update (true hierarchical)
                # If z_lower == 0 -> we keep previous state
                # Implementation detail: we will multiply update by z_lower mask
                # Combined: h_L = (z_lower * z_hard_L) * h_candidate + (1 - z_lower * z_hard_L) * h_prev
                # compute combined mask
                comb_mask = z_mask * tf.cast(z_hard_L[:, None], tf.float32)
                h_L = comb_mask * h_candidate + (1.0 - comb_mask) * h_prev

                h_states[L] = h_L
                ta_outputs[L] = ta_outputs[L].write(t_idx, h_L)
                ta_z[L] = ta_z[L].write(t_idx, tf.cast(z_hard_L, tf.int32))
                ta_z_prob[L] = ta_z_prob[L].write(t_idx, z_prob_L)

        # Stack per-layer outputs: each is [time, batch, units] -> transpose to [batch, time, units]
        layer_outputs = []
        for L in range(self.nLayers):
            stacked = ta_outputs[L].stack()  # [time, batch, units]
            stacked = tf.transpose(stacked, perm=[1, 0, 2])  # [batch, time, units]
            layer_outputs.append(stacked)

        # Concatenate along feature axis
        combined = tf.concat(layer_outputs, axis=-1)  # [batch, time, sum(units)]

        # Optional final subsampling
        if self.subsampleFactor > 1:
            combined = combined[:, ::self.subsampleFactor, :]

        # Final projection to classes
        logits = self.dense(combined)

        # Optionally return states and z signals if requested
        new_states = h_states
        # Create z tensors too (if user wants them later)
        z_tensors = [tf.transpose(ta_z[L].stack(), perm=[1, 0]) for L in range(self.nLayers)]  # [batch, time]
        zprob_tensors = [tf.transpose(ta_z_prob[L].stack(), perm=[1, 0]) for L in range(self.nLayers)]  # [batch, time]

        if returnState:
            return logits, {'states': new_states, 'z': z_tensors, 'z_prob': zprob_tensors}
        return logits

    def getSubsampledTimeSteps(self, timeSteps):
        """
        Returns predicted number of timesteps after the model's internal subsampling.
        (This preserves the API used by your decoder.)
        """
        timeSteps = tf.cast(timeSteps / self.subsampleFactor, dtype=tf.int32)
        if self.stack_kwargs is not None:
            timeSteps = tf.cast(
                (timeSteps - self.stack_kwargs['kernel_size']) / self.stack_kwargs['strides'] + 1,
                dtype=tf.int32
            )
        return timeSteps
