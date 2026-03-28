"""Masking-agnostic ResNet model for side-channel analysis.

Key design decisions informed by Marquet & Oswald (CHES 2024):
  - Multi-task: predict all bytes simultaneously (breaks the plateau)
  - Low-level parameter sharing via SharedWeightsDenseLayer (m_d design)
  - Multi-target: optionally predict BOTH s1 and t1 (exploits shared
    features between Sbox input/output without hardcoding the relationship)
  - ResNet backbone as the shared feature extractor (θ_∀)

Learnable Combiner architecture (n_components >= 2):
  Instead of predicting unmasked values directly, the model decomposes
  the prediction into n_components sub-problems (e.g., "masked value"
  and "mask"), each producing a 256-class probability distribution.
  A learnable combiner (SharedWeightsDenseLayer) discovers the
  combining operation (XOR, affine inverse, etc.) from data alone.
  This mirrors the XorLayer's decomposition structure but without
  hardcoding any specific operation — truly masking-agnostic.

No hardcoded XOR, GF256, or masking-specific operations.
"""

import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, Add, Dense,
    AveragePooling1D, Flatten, SpatialDropout1D, MultiHeadAttention,
    Softmax, Reshape, Concatenate, LayerNormalization,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import initializers, regularizers


# ---------------------------------------------------------------------------
# Adaptive input downsampling (from the reference code)
# ---------------------------------------------------------------------------

class PoolingCrop(tf.keras.layers.Layer):
    """Learnable weighted pooling for input downsampling."""

    def __init__(self, input_dim=1, name="", seed=42, **kwargs):
        if name == "":
            name = f"Crop_{np.random.randint(0, 99999)}"
        super().__init__(name=name, **kwargs)
        self.input_dim = input_dim
        self.seed = seed

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(self.input_dim, 1), dtype="float32",
            trainable=True, name=f"w_{self.name}",
            initializer=initializers.RandomUniform(seed=self.seed),
        )
        self.pooling = AveragePooling1D(pool_size=2, strides=2, padding="same")
        self.bn = BatchNormalization()
        self.dropout = tf.keras.layers.AlphaDropout(0.01)

    def call(self, inputs, training=None):
        x = tf.multiply(self.w, inputs)
        x = self.pooling(x)
        x = self.bn(x, training=training)
        x = self.dropout(x, training=training)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"input_dim": self.input_dim, "seed": self.seed})
        return config


def adaptive_downsample(x, input_length, target_size=25000, seed=42):
    """Repeatedly halve trace length until below target_size."""
    size = input_length
    it = 0
    while size > target_size:
        x = PoolingCrop(input_dim=size, seed=seed + it, name=f"ds_{it}")(x)
        size = math.ceil(size / 2)
        it += 1
    return x


# ---------------------------------------------------------------------------
# SharedWeightsDenseLayer — the paper's key innovation for m_d
# ---------------------------------------------------------------------------

class SharedWeightsDenseLayer(tf.keras.layers.Layer):
    """Dense layer with shared weights but per-branch biases.

    From Marquet & Oswald: weight matrix W is shared across all byte
    branches. Each branch gets its own bias vector, allowing byte-specific
    offsets while forcing the shared representation.

    Input shape:  (batch, features, n_branches)
    Output shape: (batch, units, n_branches)
    """

    def __init__(self, input_dim=1, units=1, shares=16,
                 activation=True, seed=42, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.units = units
        self.shares = shares
        self.use_activation = activation
        self.seed = seed

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(self.input_dim, self.units), dtype="float32",
            trainable=True, name="shared_w",
            initializer=initializers.RandomUniform(seed=self.seed),
        )
        self.b = self.add_weight(
            shape=(self.units, self.shares), dtype="float32",
            trainable=True, name="per_branch_b",
            initializer=initializers.RandomUniform(seed=self.seed),
        )

    def call(self, inputs):
        # inputs: (batch, input_dim, shares)
        x = tf.einsum("ijk,jf->ifk", inputs, self.w)
        if self.use_activation:
            return tf.keras.activations.selu(x + self.b)
        else:
            return x + self.b

    def get_config(self):
        config = super().get_config()
        config.update({
            "input_dim": self.input_dim, "units": self.units,
            "shares": self.shares, "activation": self.use_activation,
            "seed": self.seed,
        })
        return config


# ---------------------------------------------------------------------------
# ResNet backbone
# ---------------------------------------------------------------------------

def _resnet_block(x, filters, kernel_size, strides, pooling_size,
                  block_id, seed=42, l2_reg=0.0):
    """Residual block matching reference: Conv(SELU)->BN->[Pool] + skip->[Pool] -> Add."""
    shortcut = x
    reg = regularizers.l2(l2_reg) if l2_reg > 0 else None

    # Main path: Conv(SELU) -> BN -> Pool
    x = Conv1D(
        filters, kernel_size, strides=strides, padding="same",
        activation="selu",
        kernel_initializer=initializers.RandomUniform(seed=seed),
        kernel_regularizer=reg,
        name=f"res{block_id}_conv1",
    )(x)
    x = BatchNormalization(name=f"res{block_id}_bn1")(x)

    if pooling_size > 1:
        x = AveragePooling1D(pool_size=pooling_size,
                             name=f"res{block_id}_pool")(x)

    # Skip connection — must match dimensions (including pooling)
    if shortcut.shape[-1] != filters or strides > 1 or pooling_size > 1:
        shortcut = Conv1D(
            filters, 1, strides=strides, padding="same",
            kernel_initializer=initializers.RandomUniform(seed=seed),
            kernel_regularizer=reg,
            name=f"res{block_id}_skip",
        )(shortcut)
        if pooling_size > 1:
            shortcut = AveragePooling1D(pool_size=pooling_size,
                                        name=f"res{block_id}_skip_pool")(shortcut)

    x = Add(name=f"res{block_id}_add")([x, shortcut])
    return x


def build_resnet_backbone(x, num_blocks=1, filters=4, kernel_size=34,
                          strides=17, pooling_size=2, seed=42,
                          l2_reg=0.0, dropout_rate=0.0,
                          use_attention=False, attention_heads=4):
    """ResNet backbone with optional L2, dropout, and self-attention."""
    for i in range(num_blocks):
        x = _resnet_block(x, filters, kernel_size, strides, pooling_size,
                          i, seed, l2_reg=l2_reg)
        if dropout_rate > 0:
            x = SpatialDropout1D(dropout_rate, name=f"spatial_drop_{i}")(x)

    # Self-attention: lets the model learn correlations between distant
    # time points — critical for discovering second-order leakage from
    # masked implementations (mask at time t1, masked value at time t2).
    if use_attention:
        x = LayerNormalization(name="attn_ln")(x)
        attn_out = MultiHeadAttention(
            num_heads=attention_heads, key_dim=filters,
            name="self_attention",
        )(x, x)
        x = Add(name="attn_residual")([x, attn_out])

    x = Flatten(name="backbone_flat")(x)
    return x


# ---------------------------------------------------------------------------
# Prediction heads
# ---------------------------------------------------------------------------

def _per_byte_branches(backbone, n_branches, dense_units, non_shared_blocks,
                       seed, prefix="", l2_reg=0.0):
    """Create per-byte non-shared dense branches, then concatenate."""
    reg = regularizers.l2(l2_reg) if l2_reg > 0 else None
    branches = []
    for b in range(n_branches):
        x = backbone
        for blk in range(non_shared_blocks):
            x = Dense(
                dense_units, activation="selu",
                kernel_initializer=initializers.RandomUniform(seed=seed + b * 10 + blk),
                kernel_regularizer=reg,
                name=f"{prefix}branch_{b}_dense_{blk}",
            )(x)
        x = Reshape((dense_units, 1), name=f"{prefix}branch_{b}_reshape")(x)
        branches.append(x)
    return Concatenate(axis=2, name=f"{prefix}concat")(branches)


def _shared_head(concat_branches, n_branches, shared_blocks, dense_units,
                 seed, prefix=""):
    """Apply SharedWeightsDenseLayer blocks, then final 256-class output."""
    x = concat_branches
    for blk in range(shared_blocks):
        x = SharedWeightsDenseLayer(
            input_dim=x.shape[1], units=dense_units, shares=n_branches,
            activation=True, seed=seed + blk * 100,
            name=f"{prefix}shared_{blk}",
        )(x)
    # Final classification layer: 256 classes, no activation (softmax added per output)
    x = SharedWeightsDenseLayer(
        input_dim=x.shape[1], units=256, shares=n_branches,
        activation=False, seed=seed + 999,
        name=f"{prefix}shared_logits",
    )(x)
    return x


# ---------------------------------------------------------------------------
# Full model builder
# ---------------------------------------------------------------------------

def _build_component_logits(backbone, n_bytes, dense_units, non_shared_blocks,
                            shared_blocks, seed, n_components, prefix="",
                            l2_reg=0.0):
    """Build n_components sets of 256-class logits from the shared backbone.

    Each component produces (batch, 256, n_bytes) logits via its own
    per-byte branches + SharedWeightsDenseLayer head.

    Returns list of n_components tensors, each (batch, 256, n_bytes).
    """
    all_logits = []
    for c in range(n_components):
        branches = _per_byte_branches(
            backbone, n_bytes, dense_units, non_shared_blocks,
            seed + c * 2000, prefix=f"{prefix}comp{c}_",
            l2_reg=l2_reg,
        )
        logits = _shared_head(
            branches, n_bytes, shared_blocks, dense_units,
            seed + c * 2000, prefix=f"{prefix}comp{c}_",
        )
        all_logits.append(logits)
    return all_logits


class BilinearCombinerLayer(tf.keras.layers.Layer):
    """Learnable bilinear combination of two probability distributions.

    Generalizes XorLayer via low-rank CP decomposition:
        P(C=k) = Σᵢ,ⱼ P(A=i)·P(B=j)·W[i,j,k]
    where W[i,j,k] ≈ Σᵣ U[i,r]·V[j,r]·T[r,k]

    XOR is a special case: W[i,j,k] = 1 if i⊕j=k else 0.
    Affine inverse is another special case.
    The model discovers the right operation from data.

    Key property: gradients to branch A depend on branch B's distribution
    (and vice versa), creating natural pressure for complementary
    decomposition — unlike Concat→Dense which has independent gradients.

    Input:  list of 2 tensors, each (batch, n_classes, n_bytes)
    Output: (batch, n_classes, n_bytes) combined logits.
    """

    def __init__(self, rank=64, n_classes=256, shares=14, seed=42, **kwargs):
        super().__init__(**kwargs)
        self.rank = rank
        self.n_classes = n_classes
        self.shares = shares
        self.seed = seed

    def build(self, input_shape):
        init = initializers.GlorotUniform(seed=self.seed)
        # CP factors: W[i,j,k] ≈ Σᵣ U[i,r]·V[j,r]·T[r,k]
        self.U = self.add_weight(
            shape=(self.n_classes, self.rank), name="cp_U",
            initializer=init,
        )
        self.V = self.add_weight(
            shape=(self.n_classes, self.rank), name="cp_V",
            initializer=init,
        )
        self.T = self.add_weight(
            shape=(self.rank, self.n_classes), name="cp_T",
            initializer=init,
        )
        # Per-byte bias (like SharedWeightsDenseLayer)
        self.bias = self.add_weight(
            shape=(self.n_classes, self.shares), name="per_byte_bias",
            initializer="zeros",
        )

    def call(self, inputs):
        p_a, p_b = inputs  # each (batch, 256, n_bytes)

        # Project to low-rank space: (batch, rank, n_bytes)
        a_low = tf.einsum("bck,cr->brk", p_a, self.U)
        b_low = tf.einsum("bck,cr->brk", p_b, self.V)

        # Bilinear interaction — element-wise product in rank space
        product = a_low * b_low  # (batch, rank, n_bytes)

        # Project back to class space: (batch, 256, n_bytes)
        output = tf.einsum("brk,rc->bck", product, self.T)

        return output + self.bias

    def get_config(self):
        config = super().get_config()
        config.update({
            "rank": self.rank, "n_classes": self.n_classes,
            "shares": self.shares, "seed": self.seed,
        })
        return config


def _learnable_combiner(component_logits, n_bytes, combiner_rank,
                        seed, prefix="", use_skip=False):
    """Combine two component logit tensors via bilinear interaction.

    Uses BilinearCombinerLayer (low-rank CP decomposition) to learn
    the combining operation from data.

    IMPORTANT: operates on raw logits, NOT softmax probabilities.
    The previous version used Softmax(logits) → bilinear, which created
    a gradient dead zone: softmax outputs are O(1/256), so the bilinear
    product scales as O(1/256²) ≈ O(10⁻⁵), giving vanishing gradients.
    Raw logits are O(1) at initialization, giving O(1) gradient flow.

    LayerNormalization stabilizes the logit magnitudes before the
    bilinear interaction.

    Optional skip connection: output = bilinear(A, B) + A
    This gives branch A a direct gradient path, helping the backbone
    learn features even before the combiner converges.

    Returns: (batch, 256, n_bytes) combined logits.
    """
    assert len(component_logits) == 2, \
        f"Bilinear combiner requires exactly 2 components, got {len(component_logits)}"

    # LayerNorm on logits (stabilize magnitude, preserve O(1) scale)
    a = LayerNormalization(axis=1, name=f"{prefix}comp0_ln")(component_logits[0])
    b = LayerNormalization(axis=1, name=f"{prefix}comp1_ln")(component_logits[1])

    # Bilinear combination on normalized logits
    output = BilinearCombinerLayer(
        rank=combiner_rank, n_classes=256, shares=n_bytes,
        seed=seed + 9000, name=f"{prefix}bilinear_combiner",
    )([a, b])

    # Optional skip connection: output = bilinear(A, B) + A
    # Gives branch A a direct gradient path to the loss
    if use_skip:
        output = Add(name=f"{prefix}combiner_skip")([output, component_logits[0]])

    return output


def build_model(
    input_length,
    byte_range=range(2, 16),
    # ResNet backbone params
    convolution_blocks=1,
    filters=4,
    kernel_size=34,
    strides=17,
    pooling_size=2,
    # Head params
    dense_units=200,
    non_shared_blocks=1,
    shared_blocks=1,
    # Learnable combiner params
    n_components=1,
    combiner_rank=64,
    combiner_skip=False,
    # Regularization
    l2_reg=0.0,
    dropout_rate=0.0,
    # Self-attention
    use_attention=False,
    attention_heads=4,
    # Multi-target
    multi_target=False,
    # Training
    learning_rate=0.001,
    seed=42,
    summary=True,
):
    """Build the masking-agnostic multi-task ResNet model.

    Two modes controlled by n_components:

    n_components=1 (default): Direct prediction — backbone → per-byte
      branches → SharedWeights → 256-class softmax. Same as before.

    n_components=2: Bilinear Combiner — backbone → 2 sets of branches
      (each producing 256-class probabilities), then a bilinear
      combiner (low-rank CP decomposition) discovers the combining
      operation from data. Gradients couple both branches, driving
      complementary decomposition — same mechanism as XorLayer but
      without hardcoding XOR. Works for boolean, affine, or any masking.

    Args:
        input_length: Length of each power trace.
        byte_range: Which AES bytes to attack.
        convolution_blocks: Number of ResNet blocks.
        filters: Conv filters in ResNet.
        kernel_size: Conv kernel size.
        strides: Conv strides.
        pooling_size: Average pooling size after each block.
        dense_units: Units in dense head layers.
        non_shared_blocks: Non-shared dense layers per byte before shared.
        shared_blocks: SharedWeightsDenseLayer blocks.
        n_components: Number of component branches (1=direct, 2=combiner).
        combiner_rank: Rank of bilinear CP decomposition (higher=more expressive).
        combiner_skip: Add residual skip connection around the combiner.
        l2_reg: L2 regularization strength (0=off).
        dropout_rate: SpatialDropout1D rate after ResNet blocks (0=off).
        use_attention: Add multi-head self-attention after ResNet backbone.
        attention_heads: Number of attention heads.
        multi_target: If True, predict both s1 and t1.
        learning_rate: Initial learning rate.
        seed: Random seed.
        summary: Print model summary.

    Returns:
        Compiled Keras Model.
    """
    byte_list = list(byte_range)
    n_bytes = len(byte_list)

    # Input
    trace_input = Input(shape=(input_length, 1), name="traces")

    # Adaptive downsample long traces
    x = adaptive_downsample(trace_input, input_length, target_size=25000, seed=seed)

    # ResNet backbone (shared across all bytes = θ_∀)
    backbone = build_resnet_backbone(
        x, num_blocks=convolution_blocks, filters=filters,
        kernel_size=kernel_size, strides=strides,
        pooling_size=pooling_size, seed=seed,
        l2_reg=l2_reg, dropout_rate=dropout_rate,
        use_attention=use_attention, attention_heads=attention_heads,
    )

    outputs = {}
    losses = {}
    metrics = {}

    if n_components == 1:
        # Direct prediction (original architecture)
        s1_branches = _per_byte_branches(
            backbone, n_bytes, dense_units, non_shared_blocks, seed,
            prefix="s1_", l2_reg=l2_reg,
        )
        s1_logits = _shared_head(
            s1_branches, n_bytes, shared_blocks, dense_units, seed, prefix="s1_"
        )
    else:
        # Learnable Combiner: n_components branches → combiner → output
        s1_comp_logits = _build_component_logits(
            backbone, n_bytes, dense_units, non_shared_blocks,
            shared_blocks, seed, n_components, prefix="s1_",
            l2_reg=l2_reg,
        )
        s1_logits = _learnable_combiner(
            s1_comp_logits, n_bytes, combiner_rank,
            seed, prefix="s1_", use_skip=combiner_skip,
        )

    for i, b in enumerate(byte_list):
        out = Softmax(name=f"output_{b}")(s1_logits[:, :, i])
        outputs[f"output_{b}"] = out
        losses[f"output_{b}"] = "categorical_crossentropy"
        metrics[f"output_{b}"] = "accuracy"

    # Multi-target: also predict t1 (same components, separate combiner)
    if multi_target:
        if n_components == 1:
            t1_branches = _per_byte_branches(
                backbone, n_bytes, dense_units, non_shared_blocks,
                seed + 5000, prefix="t1_", l2_reg=l2_reg,
            )
            t1_logits = _shared_head(
                t1_branches, n_bytes, shared_blocks, dense_units,
                seed + 5000, prefix="t1_"
            )
        else:
            t1_comp_logits = _build_component_logits(
                backbone, n_bytes, dense_units, non_shared_blocks,
                shared_blocks, seed + 5000, n_components, prefix="t1_",
                l2_reg=l2_reg,
            )
            t1_logits = _learnable_combiner(
                t1_comp_logits, n_bytes, combiner_rank,
                seed + 5000, prefix="t1_", use_skip=combiner_skip,
            )

        for i, b in enumerate(byte_list):
            out = Softmax(name=f"output_t_{b}")(t1_logits[:, :, i])
            outputs[f"output_t_{b}"] = out
            losses[f"output_t_{b}"] = "categorical_crossentropy"
            metrics[f"output_t_{b}"] = "accuracy"

    model = Model(
        inputs={"traces": trace_input}, outputs=outputs,
        name="GeneralResNet_md",
    )
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss=losses, optimizer=optimizer, metrics=metrics)

    if summary:
        model.summary()
    return model
