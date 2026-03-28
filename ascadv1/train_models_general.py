import argparse
import os
import numpy as np
import pickle
import math
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Softmax, Dense, Input, Conv1D, AveragePooling1D, BatchNormalization, Concatenate, Add, Reshape, Lambda
from tensorflow.keras import initializers
from tensorflow.keras.optimizers import Adam
from multiprocessing import Process

# import dataset paths and variables
from utility import METRICS_FOLDER, MODEL_FOLDER

# import custom layers
from utility import XorLayer, PoolingCrop, SharedWeightsDenseLayer, GF256MultiplyLayer, GF256InvertLayer

from utility import load_dataset_multi 

import tensorflow.experimental.numpy as tnp
tnp.experimental_enable_numpy_behavior()
seed = 42

tf.random.set_seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

##############################################################
# Model: General Masking (Boolean + Multiplicative)

def general_masking(resnet = False ,convolution_blocks = 1, dense_blocks =2, kernel_size = [34],filters = 4, strides = 17 , pooling_size = 2,dense_units= 200,input_length=1000, learning_rate=0.001, classes=256 , name ='',summary = False,seed = 42):
    
    inputs_dict = {}    
    inputs  = Input(shape = (input_length,1) ,name = 'traces')
    inputs_dict['traces'] = inputs   
   
 
    branch = input_layer_creation(inputs,input_length,seed = seed)
    if resnet:
        branch = resnet_core(branch,convolution_blocks = convolution_blocks, kernel_size = kernel_size,filters = filters, strides = strides, pooling_size = pooling_size,seed = seed)
    else:
        branch = cnn_core(branch,convolution_blocks = convolution_blocks, kernel_size = kernel_size,filters = filters, strides = strides, pooling_size = pooling_size,seed = seed)

    # mask_branch = dense_core(branch,dense_blocks = dense_blocks,dense_units = dense_units,batch_norm = False,activated = True,seed = seed)
    bool_mask_branch = dense_core(branch,dense_blocks = dense_blocks,dense_units = dense_units,activated = True,seed = seed)
    mult_mask_brach = dense_core(branch,dense_blocks = dense_blocks,dense_units = dense_units,activated = True,seed = seed)
    shared_branch_t_i = dense_core_shared(branch,shared_block = 1, non_shared_block = 1, units = 200,branches = 14,seed = seed)

    # Create a learnable gate initialized to 0 (Boolean mode)
    # bias=-10 ensures sigmoid(bias) is very close to 0 at start
    gate = Dense(1, activation='sigmoid', kernel_initializer='zeros', bias_initializer=initializers.Constant(-10.0), name='mask_mode_gate')(branch)
    gate_complement = Lambda(lambda x: 1.0 - x, name='mask_mode_gate_neg')(gate)

    outputs = {} 
    metrics = {}
    for byte in range(2,16):
        
        # 1. Shared Branch Probs
        shared_branch_probs = Softmax()(shared_branch_t_i[:,:,byte-2])
        
        # 2. Boolean Unmasking Path
        xor_out = XorLayer(name ='xor_{}'.format(byte) )([shared_branch_probs, bool_mask_branch])   
        
        # 3. Affine Unmasking Path
        # Get inverse of predicted alpha
        alpha_inv = GF256InvertLayer(name='inv_alpha_{}'.format(byte))(mult_mask_brach)
        # Apply multiplicative unmasking to the result of Boolean unmasking
        unmasked_affine = GF256MultiplyLayer(name='mult_{}'.format(byte))([xor_out, alpha_inv])     
        
        # 4. Gated Fusion: Output = (1-gate)*Boolean + (gate)*Affine
        # Use tf.keras.layers.Multiply to avoid import errors
        weighted_bool = tf.keras.layers.Multiply(name='gated_bool_{}'.format(byte))([xor_out, gate_complement])
        weighted_affine = tf.keras.layers.Multiply(name='gated_affine_{}'.format(byte))([unmasked_affine, gate])
        
        final_output = Add(name='output_{}'.format(byte))([weighted_bool, weighted_affine])
        
        outputs['output_{}'.format(byte)] = final_output
        metrics['output_{}'.format(byte)] = 'accuracy'
        
    losses = {}   
 
    for k , v in outputs.items():
        losses[k] = 'categorical_crossentropy'

    model = Model(inputs = inputs_dict,outputs = outputs,name='cnn_multi_task')   
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss=losses, optimizer=optimizer, metrics=metrics) 
    if summary:
        model.summary()
    return model


################################################################
# ALL ASCAD-R EXPERIMENTS
# block of layers noted as \theta_{\forall} 

def input_layer_creation(inputs,input_length,target_size = 25000,seed = 42,name = ''):

    size = input_length
    
    iteration  = 0
    crop = inputs
    
    while size > target_size:
        crop = PoolingCrop(input_dim = size,name = name,seed = seed)(crop)
        iteration += 1
        size = math.ceil(size/2)

    x = crop  
    return x


def resnet_core(inputs_core, convolution_blocks, kernel_size, filters, strides, pooling_size, seed = 42):
    # ResNet Block adapted to match cnn_core specificities (SELU + BN order)
    # cnn_core uses: Conv(selu) -> BN -> Pool
    # We match this: Main = Conv(selu) -> BN -> [Pool]. Skip = Projection -> [Pool]. Add. 
    
    x = inputs_core #shape of x: (None, length, channels)

    # Make sure the kernel_size list matches convolution_blocks
    if len(kernel_size) != convolution_blocks:
        kernel_size = [kernel_size[0]] * convolution_blocks

    for block in range(convolution_blocks):
        shortcut = x
        
        # --- Main Branch ---
        # Match cnn_core: activation='selu', kernel_initializer=RandomUniform
        x = Conv1D(kernel_size=(kernel_size[block],), strides=strides, filters=filters, activation='selu', padding='same', kernel_initializer=tf.keras.initializers.RandomUniform(seed=seed))(x)
        x = BatchNormalization()(x)
        
        if pooling_size > 1:
            x = AveragePooling1D(pool_size=pooling_size)(x)
            
        # --- Shortcut Branch ---
        if (strides > 1) or (pooling_size > 1) or (shortcut.shape[-1] != filters):
            # Projection: Use kernel_size=1. 
            # Note: We do NOT apply SELU here usually, but since the main path is activated, 
            # we simply project linearly to match dimensions for the addition.
            shortcut = Conv1D(filters=filters, kernel_size=1, strides=strides, padding='same', kernel_initializer=tf.keras.initializers.RandomUniform(seed=seed))(shortcut)
            if pooling_size > 1:
                shortcut = AveragePooling1D(pool_size=pooling_size)(shortcut)
                
        # --- Merge ---
        x = Add()([x, shortcut])
        
    output_layer = Flatten()(x) #shape of output layer is: (None, filters * reduced_length)
    return output_layer


def cnn_core(inputs_core,convolution_blocks , kernel_size,filters, strides , pooling_size,seed = 42):
    x = inputs_core

    # Make sure the kernel_size list matches convolution_blocks
    if len(kernel_size) != convolution_blocks:
        kernel_size = [kernel_size[0]] * convolution_blocks

    for block in range(convolution_blocks):
        x = Conv1D(kernel_size=(kernel_size[block],), strides=strides, filters=filters, activation='selu', padding='same',kernel_initializer=tf.keras.initializers.RandomUniform(seed=seed))(x)    
        x = BatchNormalization()(x)
        x = AveragePooling1D(pool_size = pooling_size)(x)
        
    output_layer = Flatten()(x) 

    return output_layer


################################################################
# ALL ASCAD-R EXPERIMENTS
# prediction heads of models without low-level parameter sharing

def dense_core(inputs_core,dense_blocks,dense_units,activated = False,seed = 42):
    x = inputs_core
    for block in range(dense_blocks):
        x = Dense(dense_units,activation='selu',kernel_initializer=tf.keras.initializers.RandomUniform(seed=seed))(x)
    if activated:
        output_layer = Dense(256,activation='softmax',kernel_initializer=tf.keras.initializers.RandomUniform(seed=seed))(x)
    else:
        output_layer = Dense(256,activation=None,kernel_initializer=tf.keras.initializers.RandomUniform(seed=seed))(x)
    return output_layer    


################################################################
# ALL ASCAD-R EXPERIMENTS
# prediction heads of models with low-level parameter sharing

def dense_core_shared(inputs_core, shared_block = 1,non_shared_block = 1, units = 64, branches = 14,seed = 42):
    non_shared_branch = []
    for branch in range(branches):
        x = inputs_core
        for block in range(non_shared_block):
            x = Dense(units,activation ='selu',kernel_initializer=tf.keras.initializers.RandomUniform(seed=seed))(x)
        x_reshaped = Reshape((x.shape[1],1))(x)
        non_shared_branch.append(x_reshaped)
    x = Concatenate(axis = 2)(non_shared_branch)
   
    for block in range(shared_block):
        x = SharedWeightsDenseLayer(input_dim = x.shape[1],units = units,shares = 14,seed = seed)(x)        
    output_layer = SharedWeightsDenseLayer(input_dim = x.shape[1],units = 256,activation = False,shares = 14,seed = seed)(x)   
    return output_layer 


################################################################

class EpochSummary(tf.keras.callbacks.Callback):
    """Print one compact line per epoch: mean acc, mean val acc, loss, val loss."""

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        train_accs = [v for k, v in logs.items()
                      if k.endswith("_accuracy") and not k.startswith("val_")]
        val_accs = [v for k, v in logs.items()
                    if k.startswith("val_") and k.endswith("_accuracy")]
        train_loss = logs.get("loss", float("nan"))
        val_loss = logs.get("val_loss", float("nan"))
        mean_acc = np.mean(train_accs) if train_accs else float("nan")
        mean_val_acc = np.mean(val_accs) if val_accs else float("nan")
        lr = float(self.model.optimizer.learning_rate)
        print(f"Epoch {epoch + 1:3d} | "
              f"loss: {train_loss:.4f}  acc: {mean_acc:.4f} | "
              f"val_loss: {val_loss:.4f}  val_acc: {mean_val_acc:.4f} | "
              f"lr: {lr:.6f}")

class WarmupReduceLROnPlateau(tf.keras.callbacks.Callback):
    """
    Wrapper around tf.keras.callbacks.ReduceLROnPlateau that only activates
    after a warm-up period (start_from_epoch). This mirrors EarlyStopping's
    start_from_epoch behavior so LR reduction begins at the same point.
    """
    def __init__(self, start_from_epoch=40, **rlrop_kwargs):
        super().__init__()
        self.start_from_epoch = start_from_epoch
        self._rlrop = tf.keras.callbacks.ReduceLROnPlateau(**rlrop_kwargs)

    def set_model(self, model):
        self._rlrop.set_model(model)

    def set_params(self, params):
        self._rlrop.set_params(params)

    def on_train_begin(self, logs=None):
        self._rlrop.on_train_begin(logs)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) >= self.start_from_epoch:
            self._rlrop.on_epoch_end(epoch, logs)

#### Training high level function #####
def train_model(training_type,byte,seed,model_name=None, epochs = 100, resnet = False, learning_rate=0.001, batch_size=250, convolution_blocks=1, pooling_size=2, filters=4, strides=17, dropout_rate=0.3, regularization_factor=1e-4, early_stopping = False, reduce_lr=False, lr_reduce_factor=0.1):
    seed = int(seed)
    epochs = epochs
    batch_size = batch_size
    n_traces = 50000
    convolution_blocks = convolution_blocks

    # Load dataset for general masking model
    X_profiling , validation_data = load_dataset_multi('t1',n_traces = n_traces,dataset = 'training')
    model_t = 'model_{}'.format(training_type)

    window =  X_profiling.element_spec[0]['traces'].shape[0]
    monitor = 'val_loss'
    mode = 'min'
    warmup_start_epoch = 40  # align LR reduction start with EarlyStopping

    # Print training summary (GeneralArch style)
    print(f"=== Training {training_type} | seed={seed} ===")
    print(f"Bytes: 2-15 | ResNet: {resnet}")
    lr_mode = "static" if not reduce_lr else "ReduceLR"
    es_mode = f"patience=20, warmup={warmup_start_epoch}" if early_stopping else "off"
    print(f"Traces: {n_traces} | Epochs: {epochs}")
    print(f"LR: {learning_rate} ({lr_mode}) | Early stopping: {es_mode}")
    reg_parts = []
    if regularization_factor > 0:
        reg_parts.append(f"L2={regularization_factor}")
    if dropout_rate > 0:
        reg_parts.append(f"dropout={dropout_rate}")
    if reg_parts:
        print(f"Regularization: {', '.join(reg_parts)}")

    # Create the general masking model
    model = general_masking(
        input_length=window,
        seed=seed,
        resnet=resnet,
        learning_rate=learning_rate,
        convolution_blocks=convolution_blocks,
        pooling_size=pooling_size,
        filters=filters,
        strides=strides
    )

    X_profiling = X_profiling.shuffle(len(X_profiling)).batch(batch_size)
    validation_data = validation_data.batch(batch_size)

    file_name = '{}_{}'.format(model_name,byte) if model_name else '{}_{}'.format(model_t,byte)
    
    # Optional: Keras EarlyStopping with warm-up to avoid premature stopping
    if early_stopping:
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            mode=mode,
            min_delta=1e-3,
            patience=20,
            start_from_epoch=warmup_start_epoch,
            restore_best_weights=True,
            verbose=1,
        )
    # Optional: Reduce LR on plateau (acts at same warm-up point as EarlyStopping)
    if reduce_lr and not early_stopping:
        reduce_lr_cb = WarmupReduceLROnPlateau(
            start_from_epoch=warmup_start_epoch,
            monitor=monitor,
            mode=mode,
            factor=lr_reduce_factor,
            patience=20,
            min_delta=1e-3,
            cooldown=0,
            min_lr=1e-6,
            verbose=1,
        )

    ckpt_path = MODEL_FOLDER + file_name + '.weights.h5'
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=ckpt_path,
            save_weights_only=True,
            monitor=monitor,
            mode=mode,
            save_best_only=True,
            verbose=1),
        EpochSummary()
    ]
    if early_stopping:
        callbacks.append(early_stop)
    elif reduce_lr:
        callbacks.append(reduce_lr_cb)
    
    history = model.fit(X_profiling, batch_size=batch_size, verbose=0, epochs=epochs, validation_data=validation_data, callbacks=callbacks)

    print(f"Best  weights: {ckpt_path}")

    hist_path = METRICS_FOLDER + 'history_training_' + file_name
    file = open(hist_path, 'wb')
    pickle.dump(history.history, file)
    file.close()
    print(f"History: {hist_path}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trains Neural Network Models')
    parser.add_argument("--model_name", type=str, default='general_masking', help="Name of the model to be trained")
    parser.add_argument("--seed", type=int, default=None, help="Seed to be used")
    parser.add_argument("--training_type", type=str, default='general_masking', help="Training type")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--resnet", action='store_true', default=False, help="Use ResNet")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=250, help="Batch size")
    parser.add_argument("--convolution_blocks", type=int, default=1, help="Number of convolution blocks")
    parser.add_argument("--pooling_size", type=int, default=2, help="Pooling size")
    parser.add_argument("--filters", type=int, default=4, help="Number of filters")
    parser.add_argument("--strides", type=int, default=17, help="Strides")
    parser.add_argument("--dropout_rate", type=float, default=0.3, help="Dropout rate for L2 models")
    parser.add_argument("--regularization_factor", type=float, default=1e-4, help="L2 regularization factor for L2 models")
    parser.add_argument("--early_stopping", action='store_true', default=False, help="Enable EarlyStopping with warm-up")
    parser.add_argument("--reduce_lr", action='store_true', default=False, help="Enable ReduceLROnPlateau (warm-up aligned with EarlyStopping)")
    parser.add_argument("--lr_reduce_factor", type=float, default=0.1, help="LR reduction factor (e.g., 0.1 to divide by 10)")

    args            = parser .parse_args()
  
    model_name_arg = args.model_name
    training_type_arg = args.training_type
    seed = args.seed
    epochs = args.epochs
    resnet = args.resnet
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    convolution_blocks = args.convolution_blocks
    pooling_size = args.pooling_size
    filters = args.filters
    strides = args.strides
    dropout_rate = args.dropout_rate
    regularization_factor = args.regularization_factor
    early_stopping = args.early_stopping
    reduce_lr = args.reduce_lr
    lr_reduce_factor = args.lr_reduce_factor
     
    training_types = [training_type_arg]

    seeds_random = np.random.randint(0,9999,size = 9)
    seeds_random = [seed] if seed else np.concatenate([[42],seeds_random],axis = 0)
    
    for seed in seeds_random:
        tf.random.set_seed(seed)
        np.random.seed(seed)
        for training_type in training_types:
            
            if early_stopping and reduce_lr:
                print("Warning: Both --early_stopping and --reduce_lr set. Prioritizing early stopping and ignoring LR reduction.")
            process_eval = Process(target=train_model, args=(training_type,'all',seed, model_name_arg, epochs, resnet, learning_rate, batch_size, convolution_blocks, pooling_size, filters, strides, dropout_rate, regularization_factor, early_stopping, reduce_lr and not early_stopping, lr_reduce_factor))
            process_eval.start()
            process_eval.join()      

    print("$ Done !")



