import argparse
import os
import numpy as np
import pickle
import math
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten,Softmax, Dense, Input,Conv1D, AveragePooling1D, BatchNormalization , Concatenate, Add, ReLU, Reshape, LayerNormalization, Dropout, SpatialDropout1D, Lambda, Permute
from tensorflow.keras import initializers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from multiprocessing import Process


# import dataset paths and variables
from utility import  METRICS_FOLDER , MODEL_FOLDER

# import custom layers
from utility import XorLayer , PoolingCrop  , SharedWeightsDenseLayer, GF256MultiplyLayer, GF256InvertLayer

from utility import load_dataset, load_dataset_multi 

import tensorflow.experimental.numpy as tnp
tnp.experimental_enable_numpy_behavior()
seed = 42


tf.random.set_seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

  

###########################################################################
# ALL ASCAD-R experiments 
#   model single-task m_s

def model_single_task(resnet = False ,convolution_blocks = 1,dense_blocks =2, kernel_size = [34],filters = 4, strides = 17 , pooling_size = 2,dense_units= 200,input_length=1000, learning_rate=0.001, classes=256 , name ='',summary = True,seed = 42):
    inputs_dict = {}
    
    inputs  = Input(shape = (input_length,1) ,name = 'traces')
    inputs_dict['traces'] = inputs   
    
    branch = input_layer_creation(inputs,input_length,seed = seed)
    if resnet:
        branch = resnet_core(branch,convolution_blocks = 1, kernel_size = kernel_size,filters = filters, strides = strides, pooling_size = pooling_size,seed = seed)
    else:
        branch = cnn_core(branch,convolution_blocks = 1, kernel_size = kernel_size,filters = filters, strides = strides, pooling_size = pooling_size,seed = seed)

    outputs = {}
      
    mask_branch = dense_core(branch,dense_blocks = dense_blocks,dense_units = dense_units,activated = True,seed = seed)
    intermediate_branch = dense_core(branch,dense_blocks = dense_blocks,dense_units = dense_units,activated = True,seed = seed)
    
    xor = XorLayer(name ='output' )([mask_branch,intermediate_branch]) 
    outputs['output'] = xor
    
    optimizer = Adam(learning_rate=learning_rate)
    model = Model(inputs = inputs_dict,outputs = outputs,name='cnn_single_task_xor_{}'.format(name))

    
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    if summary:
        model.summary() 
    return model


##############################################################
#  Section 5.1 Leveraging common masks with a shared branch : 
#   model multi-task m_{nt + d + 1} 

def model_multi_task_single_target_one_shared_mask(resnet = False ,convolution_blocks = 1,dense_blocks =2, kernel_size = [34],filters = 4, strides = 17 , pooling_size = 2,dense_units= 200,input_length=1000, learning_rate=0.001, classes=256 , name ='',summary = False,seed = 42):
    
    inputs_dict = {}    
    outputs = {} 
    metrics = {}
    inputs  = Input(shape = (input_length,1) ,name = 'traces')
    inputs_dict['traces'] = inputs   
   
 
    branch = input_layer_creation(inputs,input_length,seed = seed)
    if resnet:
        branch = resnet_core(branch,convolution_blocks = convolution_blocks, kernel_size = kernel_size,filters = filters, strides = strides, pooling_size = pooling_size,seed = seed)
    else:
        branch = cnn_core(branch,convolution_blocks = convolution_blocks, kernel_size = kernel_size,filters = filters, strides = strides, pooling_size = pooling_size,seed = seed)

    # mask_branch = dense_core(branch,dense_blocks = dense_blocks,dense_units = dense_units,batch_norm = False,activated = True,seed = seed)
    mask_branch = dense_core(branch,dense_blocks = dense_blocks,dense_units = dense_units,activated = True,seed = seed)



    

    for byte in range(2,16):
        
        # intermediate_branch = dense_core(branch,dense_blocks = dense_blocks,dense_units = dense_units,batch_norm = False,activated = False,seed = seed)
        intermediate_branch = dense_core(branch,dense_blocks = dense_blocks,dense_units = dense_units,activated = False,seed = seed)
        xor = XorLayer(name ='xor_{}'.format(byte) )([intermediate_branch,mask_branch])        
        outputs['output_{}'.format(byte)] = Softmax(name = 'output_{}'.format(byte))(xor)
        metrics['output_{}'.format(byte)] = 'accuracy'
        
    losses = {}   
    




    for k , v in outputs.items():
        losses[k] = 'categorical_crossentropy'


    model = Model(inputs = inputs_dict,outputs = outputs,name='cnn_multi_task')        

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss=losses, optimizer=optimizer,metrics = metrics)
    
    if summary:
        model.summary()
    return model   
##############################################################
#  Section 5.1 Leveraging common masks with a shared branch : 
#   model multi-task m_{d} 

def model_multi_task_single_target_one_shared_mask_shared_branch(resnet = False ,convolution_blocks = 1,dense_blocks =2, kernel_size = [34],filters = 4, strides = 17 , pooling_size = 2,dense_units= 200,input_length=1000, learning_rate=0.001, classes=256 , name ='',summary = False,seed = 42):
    
    inputs_dict = {}    
    inputs  = Input(shape = (input_length,1) ,name = 'traces')
    inputs_dict['traces'] = inputs   
   
 
    branch = input_layer_creation(inputs,input_length,seed = seed)
    if resnet:
        branch = resnet_core(branch,convolution_blocks = convolution_blocks, kernel_size = kernel_size,filters = filters, strides = strides, pooling_size = pooling_size,seed = seed)
    else:
        branch = cnn_core(branch,convolution_blocks = convolution_blocks, kernel_size = kernel_size,filters = filters, strides = strides, pooling_size = pooling_size,seed = seed)

    # mask_branch = dense_core(branch,dense_blocks = dense_blocks,dense_units = dense_units,batch_norm = False,activated = True,seed = seed)
    mask_branch = dense_core(branch,dense_blocks = dense_blocks,dense_units = dense_units,activated = True,seed = seed)
    shared_branch_t_i = dense_core_shared(branch,shared_block = 1, non_shared_block = 1, units = 200,branches = 14,seed = seed)

    outputs = {} 

    metrics = {}
    for byte in range(2,16):
        
        xor = XorLayer(name ='xor_{}'.format(byte) )([shared_branch_t_i[:,:,byte-2],mask_branch])        
        outputs['output_{}'.format(byte)] = Softmax(name = 'output_{}'.format(byte))(xor)
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


##############################################################
#   model multi-task m_{d} wihtout xor layer:

def model_multi_task_single_target_one_shared_mask_shared_branch_no_xor(resnet = False ,convolution_blocks = 1,dense_blocks =2, kernel_size = [34],filters = 4, strides = 5 , pooling_size = 1, dense_units= 200,input_length=1000, learning_rate=0.001, classes=256 , name ='',summary = False,seed = 42, dropout_rate=0.3, regularization_factor=1e-4):
    
    inputs_dict = {}    
    inputs  = Input(shape = (input_length,1) ,name = 'traces')
    inputs_dict['traces'] = inputs   
   
    
    # # Make sure the kernel_size list matches convolution_blocks
    # if len(kernel_size) != convolution_blocks:
    #     kernel_size = [kernel_size[0]] * convolution_blocks

    branch = input_layer_creation(inputs,input_length,seed = seed)
    if resnet:
        branch = resnet_core_L2(branch,convolution_blocks = convolution_blocks, kernel_size = kernel_size,filters = filters, strides = strides, pooling_size = pooling_size,seed = seed, regularization_factor=regularization_factor, dropout_rate=dropout_rate)
        # branch = resnet_core(branch,convolution_blocks = 2, kernel_size = kernel_size,filters = filters, strides = strides, pooling_size = pooling_size,seed = seed)
    else:
        branch = cnn_core(branch,convolution_blocks = convolution_blocks, kernel_size = kernel_size,filters = filters, strides = strides, pooling_size = pooling_size,seed = seed)

    # mask_branch = dense_core(branch,dense_blocks = dense_blocks,dense_units = dense_units,batch_norm = False,activated = True,seed = seed)
    # mask_branch = dense_core(branch,dense_blocks = dense_blocks,dense_units = dense_units,activated = True,seed = seed)
    shared_branch_t_i = dense_core_shared_L2(branch,shared_block = 1, non_shared_block = 1, units = 200,branches = 14,seed = seed, dropout_rate=dropout_rate, regularization_factor=regularization_factor)

    outputs = {} 

    metrics = {}
    for byte in range(2,16):
        
        # xor = XorLayer(name ='xor_{}'.format(byte) )([shared_branch_t_i[:,:,byte-2],mask_branch])        
        outputs['output_{}'.format(byte)] = Softmax(name = 'output_{}'.format(byte))(shared_branch_t_i[:,:,byte-2])
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


def model_multi_task_single_target_one_shared_mask_shared_branch_Transformer_branch(resnet = False ,convolution_blocks = 1,dense_blocks =2, kernel_size = [34],filters = 4, strides = 17 , pooling_size = 2,dense_units= 200,input_length=1000, learning_rate=0.001, classes=256 , name ='',summary = False,seed = 42):
    
    inputs_dict = {}    
    inputs  = Input(shape = (input_length,1) ,name = 'traces')
    inputs_dict['traces'] = inputs   
   
 
    branch = input_layer_creation(inputs,input_length,seed = seed)
    if resnet:
        branch = resnet_core(branch,convolution_blocks = convolution_blocks, kernel_size = kernel_size,filters = filters, strides = strides, pooling_size = pooling_size,seed = seed)
    else:
        branch = cnn_core(branch,convolution_blocks = convolution_blocks, kernel_size = kernel_size,filters = filters, strides = strides, pooling_size = pooling_size,seed = seed)

    # mask_branch = dense_core(branch,dense_blocks = dense_blocks,dense_units = dense_units,batch_norm = False,activated = True,seed = seed)
    # mask_branch = dense_core(branch,dense_blocks = dense_blocks,dense_units = dense_units, activated = True, seed = seed)

    # insert fetures into a small transformer to analyze relationships between bytes, strip them from the mask if any,
    # and input the output to the shared-weigths dense layers
    # the transformer will output the same shape as the resnet / cnn core output

    # print ("Branch shape before transformer: ", branch.shape)
    transformer_branch = transformer_block(branch, num_heads=4, ff_dim=512, dropout=0.1, seed=seed, name_prefix="trfmer")
    # print("Transformer branch shape: ", transformer_branch.shape)
    # shared_branch_t_i = dense_core_shared(branch, shared_block = 1, non_shared_block = 1, units = 200,branches = 14,seed = seed)

    shared_branch_t_i = dense_core_shared(transformer_branch,shared_block = 1, non_shared_block = 1, units = 200,branches = 14,seed = seed)

    outputs = {} 

    metrics = {}
    for byte in range(2,16):
        
        # xor = XorLayer(name ='xor_{}'.format(byte) )([shared_branch_t_i[:,:,byte-2],mask_branch])        
        # outputs['output_{}'.format(byte)] = Softmax(name = 'output_{}'.format(byte))(xor)
        outputs['output_{}'.format(byte)] = Softmax(name = 'output_{}'.format(byte))(shared_branch_t_i[:,:,byte-2])
        metrics['output_{}'.format(byte)] = 'accuracy'
        
    losses = {}   
    



    for k , v in outputs.items():
        losses[k] = 'categorical_crossentropy'


    # model = Model(inputs = inputs_dict,outputs = outputs,name='cnn_multi_task')   
    model = Model(inputs = inputs_dict,outputs = outputs,name='ResNet_multi_task_Transformer_branch')
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss=losses, optimizer=optimizer, metrics=metrics) 
    if summary:
        model.summary()
    return model  


##############################################################
#  Section 5.2 Leveraging different masks using low-level parameter sharing : 
#   model multi-task m_{d} 

def model_multi_task_single_target(resnet = False ,convolution_blocks = 1,dense_blocks =2, kernel_size = [34],filters = 4, strides = 17 , pooling_size = 2,dense_units= 200,input_length=1000, learning_rate=0.001, classes=256 , name ='',summary = False,seed = 42):
    
    inputs_dict = {}    
    inputs  = Input(shape = (input_length,1) ,name = 'traces')
    inputs_dict['traces'] = inputs   
   
 
    branch = input_layer_creation(inputs,input_length,seed = seed)
    if resnet:
        branch = resnet_core(branch,convolution_blocks = convolution_blocks, kernel_size = kernel_size,filters = filters, strides = strides, pooling_size = pooling_size,seed = seed)
    else:
        branch = cnn_core(branch,convolution_blocks = convolution_blocks, kernel_size = kernel_size,filters = filters, strides = strides, pooling_size = pooling_size,seed = seed)

    shared_branch_s_r = dense_core_shared(branch,shared_block = 1, non_shared_block = 1, units = 200,branches = 14,seed = seed)
    shared_branch_r = dense_core_shared(branch,shared_block = 1, non_shared_block = 1, units = 200,branches = 14,seed = seed)
    
    outputs = {} 
    
    for byte in range(2,16):
        
        shared_branch_r_activated = Softmax()(shared_branch_r[:,:,byte-2])   
        
        xor = XorLayer(name ='xor_{}'.format(byte) )([shared_branch_s_r[:,:,byte-2],shared_branch_r_activated])        
        outputs['output_{}'.format(byte)] = Softmax(name = 'output_{}'.format(byte))(xor)
        
    losses = {}   
    



    for k , v in outputs.items():
        losses[k] = 'categorical_crossentropy'


    model = Model(inputs = inputs_dict,outputs = outputs,name='cnn_multi_task')      
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss=losses, optimizer=optimizer, metrics=['accuracy']) 
    if summary:
        model.summary()
    return model   

##############################################################
#  Section 5.2 Leveraging different masks using low-level parameter sharing : 
#   model multi-task m_{nt * d} 

def model_multi_task_single_target_not_shared(resnet = False ,convolution_blocks = 1,dense_blocks =2, kernel_size = [34],filters = 4, strides = 17 , pooling_size = 2,dense_units= 200,input_length=1000, learning_rate=0.001, classes=256 , name ='',summary = False,seed = 42):
    
    inputs_dict = {}    
    inputs  = Input(shape = (input_length,1) ,name = 'traces')
    inputs_dict['traces'] = inputs   
   
 
    branch = input_layer_creation(inputs,input_length,seed = seed)
    if resnet:
        branch = resnet_core(branch,convolution_blocks = convolution_blocks, kernel_size = kernel_size,filters = filters, strides = strides, pooling_size = pooling_size,seed = seed)
    else:
        branch = cnn_core(branch,convolution_blocks = convolution_blocks, kernel_size = kernel_size,filters = filters, strides = strides, pooling_size = pooling_size,seed = seed)


    outputs = {} 
    
    for byte in range(2,16):
        
        # intermediate_branch = dense_core(branch,dense_blocks = dense_blocks,dense_units = dense_units,batch_norm = False,activated = False,seed = seed)
        # mask_branch = dense_core(branch,dense_blocks = dense_blocks,dense_units = dense_units,batch_norm = False,activated = True,seed = seed)
        intermediate_branch = dense_core(branch,dense_blocks = dense_blocks,dense_units = dense_units,activated = False,seed = seed)
        mask_branch = dense_core(branch,dense_blocks = dense_blocks,dense_units = dense_units,activated = True,seed = seed)
                
        xor = XorLayer(name ='xor_{}'.format(byte) )([intermediate_branch,mask_branch])        
        outputs['output_{}'.format(byte)] = Softmax(name = 'output_{}'.format(byte))(xor)
    losses = {}   
    



    for k , v in outputs.items():
        losses[k] = 'categorical_crossentropy'


    model = Model(inputs = inputs_dict,outputs = outputs,name='cnn_multi_task')    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss=losses, optimizer=optimizer, metrics=['accuracy']) 
    if summary:
        model.summary()
    return model   

##############################################################
#  Section 5.3 Leveraging different targets masked by the same randomness : 
#   model multi-task m_{d} for Affine Masking (ASCADv2)
#   Z = (S ^ beta) * alpha

def model_multi_task_affine(resnet = False ,convolution_blocks = 1,dense_blocks =2, kernel_size = [34],filters = 4, strides = 17 , pooling_size = 2,dense_units= 200,input_length=1000, learning_rate=0.001, classes=256 , name ='',summary = False,seed = 42, shared_branch = False):
    
    inputs_dict = {}    
    outputs = {} 
    metrics = {}
    inputs  = Input(shape = (input_length,1) ,name = 'traces')
    inputs_dict['traces'] = inputs   
   
    # Main branch for shared feature extraction
    branch = input_layer_creation(inputs,input_length,seed = seed)
    if resnet:
        branch = resnet_core(branch,convolution_blocks = convolution_blocks, kernel_size = kernel_size,filters = filters, strides = strides, pooling_size = pooling_size,seed = seed)
    else:
        branch = cnn_core(branch,convolution_blocks = convolution_blocks, kernel_size = kernel_size,filters = filters, strides = strides, pooling_size = pooling_size,seed = seed)

    # 1. Alpha Branch (Multiplicative Mask)
    alpha_branch = dense_core(branch, dense_blocks=dense_blocks, dense_units=dense_units, activated=True, seed=seed)
    outputs['alpha'] = Softmax(name='alpha')(alpha_branch) 
    metrics['alpha'] = 'accuracy'

    # 2. Beta Branch (Additive Mask)
    beta_branch = dense_core(branch, dense_blocks=dense_blocks, dense_units=dense_units, activated=True, seed=seed)
    outputs['beta'] = Softmax(name='beta')(beta_branch) 
    metrics['beta'] = 'accuracy'

    # 3. Rin Branch (Random Input Mask)
    rin_branch = dense_core(branch, dense_blocks=dense_blocks, dense_units=dense_units, activated=True, seed=seed)
    outputs['rin'] = Softmax(name='rin')(rin_branch)
    metrics['rin'] = 'accuracy'
    
    # 3. Share/S-box Output Branch - One per byte
    # This predicts the UNMASKED sensitive value S (s1)
    # This matches the label 's1' loaded by default in load_dataset_multi('s1')
    
    if shared_branch:
        # Use shared weights for the S-branches
        # Assuming units=200 matches dense_units default
        s_branches_shared = dense_core_shared(branch, shared_block=1, non_shared_block=1, units=dense_units, branches=14, seed=seed)
        
    for byte in range(2, 16):
        # Head for Share S_byte
        if shared_branch:
            # Slice the shared output. 
            # Note: dense_core_shared uses SharedWeightsDenseLayer which outputs (Batch, Units?, Shares)
            # Actually dense_core_shared ends with SharedWeightsDenseLayer(..., output_units=256 ...)
            # So output is (Batch, 256, 14).
            # We slice [:, :, byte-2] -> (Batch, 256)
            # But wait, dense_core_shared returns unactivated output if activation=False
            # Let's check dense_core_shared implementation: 
            # output_layer = SharedWeightsDenseLayer(..., activation = False, ...)(x)
            # So we apply Softmax here.
            s_branch = s_branches_shared[:, :, byte-2]
        else:
            s_branch = dense_core(branch, dense_blocks=dense_blocks, dense_units=dense_units, activated=True, seed=seed)
        
        # We name this 'output_{}' to match the primary target expected by metrics/callbacks
        # We name this 'output_{}' to match the primary target expected by metrics/callbacks
        outputs['output_{}'.format(byte)] = Softmax(name='output_{}'.format(byte))(s_branch)
        metrics['output_{}'.format(byte)] = 'accuracy'
        
    losses = {}   
    # Alpha/Beta might not be available in validation loop or metrics unless we handle it?
    # But compile needs loss for every output.
    for k , v in outputs.items():
        losses[k] = 'categorical_crossentropy'

    model = Model(inputs = inputs_dict,outputs = outputs,name='cnn_multi_task_affine')        

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss=losses, optimizer=optimizer,metrics = metrics)
    
    if summary:
        model.summary()
    return model   


# ############################################################
# Transformer core

def transformer_block(x, num_heads=4, ff_dim=512, dropout=0.1, seed=42, name_prefix="tr"):
    # x: (batch, seq_len, d_model)
    d_model = x.shape[-1]
    attn = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model // num_heads,
        dropout=dropout,
        kernel_initializer=initializers.GlorotUniform(seed=seed),
        name=f"{name_prefix}_mha"
    )(x, x)
    x = LayerNormalization(epsilon=1e-5, name=f"{name_prefix}_ln1")(x + attn)
    ff = Dense(ff_dim, activation="selu",
               kernel_initializer=initializers.GlorotUniform(seed=seed+1),
               name=f"{name_prefix}_ff1")(x)
    ff = Dropout(dropout, seed=seed+2, name=f"{name_prefix}_do1")(ff)
    ff = Dense(d_model,
               kernel_initializer=initializers.GlorotUniform(seed=seed+3),
               name=f"{name_prefix}_ff2")(ff)
    x = LayerNormalization(epsilon=1e-5, name=f"{name_prefix}_ln2")(x + ff)
    return x

def model_multi_task_shared_branch_transformer(resnet=False, convolution_blocks=1, dense_blocks=2, kernel_size=[34], filters=4, strides=17, pooling_size=2, dense_units=200, input_length=1000, learning_rate=0.001, classes=256, name='', summary=False, seed=42, num_heads=4, ff_dim=512, num_layers=2, dropout=0.1):
    inputs_dict = {}
    inputs = Input(shape=(input_length, 1), name='traces')
    inputs_dict['traces'] = inputs

    # Backbone
    branch = input_layer_creation(inputs, input_length, seed=seed)
    if resnet:
        branch = resnet_core(branch, convolution_blocks, kernel_size, filters, strides, pooling_size, seed=seed)
    else:
        branch = cnn_core(branch, convolution_blocks, kernel_size, filters, strides, pooling_size, seed=seed)

    # Mask token (learned mask-like representation)
    mask_token = dense_core(branch, dense_blocks=dense_blocks, dense_units=dense_units,
                            activated=True, seed=seed)                # (batch, d_model)
    mask_token = Lambda(lambda x: tf.expand_dims(x, axis=1), name="mask_token_expand")(mask_token)  # (batch, 1, d_model)

    # Byte features (shared low-level + per-byte heads)
    shared_branch_t_i = dense_core_shared(branch, shared_block=1, non_shared_block=1,
                                          units=200, branches=14, seed=seed)  # (batch, 14, d_model)

    # Build token sequence: [mask_token] + 14 byte tokens
    shared_branch_t_i = Permute((2, 1))(shared_branch_t_i)  # Swap last 2 dims: (batch, 256, 14) → (batch, 14, 256)
    tokens = Concatenate(axis=1, name="tokens_concat")([mask_token, shared_branch_t_i])  # (batch, 15, d_model)

    # Transformer stack
    x = tokens
    for layer_idx in range(num_layers):
        x = transformer_block(
            x,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            seed=seed + layer_idx * 10,
            name_prefix=f"tr_block_{layer_idx}"
        )

    # Drop the mask token; keep byte tokens
    byte_tokens = x[:, 1:, :]  # (batch, 14, d_model)

    outputs = {}
    metrics = {}
    for idx, byte in enumerate(range(2, 16)):  # bytes 2..15
        logits = Dense(classes,
                       kernel_initializer=initializers.GlorotUniform(seed=seed + 100 + byte),
                       name=f"output_{byte}_logits")(byte_tokens[:, idx, :])
        outputs[f"output_{byte}"] = Softmax(name=f"output_{byte}")(logits)
        metrics[f"output_{byte}"] = "accuracy"

    losses = {k: "categorical_crossentropy" for k in outputs.keys()}

    model = Model(inputs=inputs_dict, outputs=outputs,
                  name='cnn_multi_task_transformer_unmask')

    model.compile(loss=losses, optimizer=Adam(learning_rate=learning_rate), metrics=metrics)
    if summary:
        model.summary()
    return model

#################################################################
# Boolean Mask & Multiplicative Mask Layers


def model_multi_task_single_target_one_shared_mask_shared_branch_general_masking_l2(resnet = False ,convolution_blocks = 1,dense_blocks =2, kernel_size = [34],filters = 4, strides = 17 , pooling_size = 2,dense_units= 200,input_length=1000, learning_rate=0.001, classes=256 , name ='',summary = False,seed = 42, regularization_factor = 0.01):
    
    inputs_dict = {}    
    inputs  = Input(shape = (input_length,1) ,name = 'traces')
    inputs_dict['traces'] = inputs   
   
 
    branch = input_layer_creation(inputs,input_length,seed = seed)
    if resnet:
        branch = resnet_core(branch,convolution_blocks = convolution_blocks, kernel_size = kernel_size,filters = filters, strides = strides, pooling_size = pooling_size,seed = seed)
    else:
        branch = cnn_core(branch,convolution_blocks = convolution_blocks, kernel_size = kernel_size,filters = filters, strides = strides, pooling_size = pooling_size,seed = seed)

    # mask_branch = dense_core(branch,dense_blocks = dense_blocks,dense_units = dense_units,batch_norm = False,activated = True,seed = seed)
    # bool_mask_branch = dense_core(branch,dense_blocks = dense_blocks,dense_units = dense_units,activated = True,seed = seed)
    bool_mask_branch = dense_core_l2(branch,dense_blocks = dense_blocks,dense_units = dense_units,activated = True,seed = seed,regularization_factor = regularization_factor)
    mult_mask_brach = dense_core_l2(branch,dense_blocks = dense_blocks,dense_units = dense_units,activated = True,seed = seed,regularization_factor = regularization_factor)
    # shared_branch_t_i = dense_core_shared(branch,shared_block = 1, non_shared_block = 1, units = 200,branches = 14,seed = seed)
    shared_branch_t_i = dense_core_shared(branch,shared_block = 1, non_shared_block = 1, units = 200,branches = 14,seed = seed,regularization_factor = regularization_factor)

    outputs = {} 

    metrics = {}
    for byte in range(2,16):
        shared_branch_probs = Softmax()(shared_branch_t_i[:,:,byte-2])
        xor_out = XorLayer(name ='xor_{}'.format(byte) )([shared_branch_probs, bool_mask_branch])   
        alpha_inv = GF256InvertLayer(name='inv_alpha_{}'.format(byte))(mult_mask_brach)    
        unmasked = GF256MultiplyLayer(name='output_{}'.format(byte))([xor_out, alpha_inv])     
        outputs['output_{}'.format(byte)] = unmasked
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


def model_multi_task_single_target_one_shared_mask_shared_branch_general_masking(resnet = False ,convolution_blocks = 1,dense_blocks =2, kernel_size = [34],filters = 4, strides = 17 , pooling_size = 2,dense_units= 200,input_length=1000, learning_rate=0.001, classes=256 , name ='',summary = False,seed = 42):
    
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
        ############# DEBUG ###########
        # print('input_layer_creation - iteration {}: size {}'.format(iteration,size))
        ############# DEBUG ###########
        iteration += 1
        size = math.ceil(size/2)

    x = crop  
    return x


# def resnet_core(inputs_core, convolution_blocks, kernel_size, filters, strides, pooling_size, seed = 42):
#     # ResNet Block adapted to match cnn_core specificities (SELU + BN order)
#     # cnn_core uses: Conv(selu) -> BN -> Pool
#     # We match this: Main = Conv(selu) -> BN -> [Pool]. Skip = Projection -> [Pool]. Add. 
    
#     x = inputs_core #shape of x: (None, length, channels)
#     for block in range(convolution_blocks):
#         shortcut = x
        
#         # --- Main Branch ---
#         # Match cnn_core: activation='selu', kernel_initializer=RandomUniform
#         x = Conv1D(kernel_size=(kernel_size[block],), strides=strides, filters=filters, activation='selu', padding='same', kernel_initializer=tf.keras.initializers.RandomUniform(seed=seed))(x)
#         x = BatchNormalization()(x)
#         # x = ReLU()(x)  <-- Removed to match cnn_core's activation logic (integrated in Conv)
        
#         if pooling_size > 1:
#             x = AveragePooling1D(pool_size=pooling_size)(x)
            
#         # --- Shortcut Branch ---
#         if (strides > 1) or (pooling_size > 1) or (shortcut.shape[-1] != filters):
#             # Projection: Use kernel_size=1. 
#             # Note: We do NOT apply SELU here usually, but since the main path is activated, 
#             # we simply project linearly to match dimensions for the addition.
#             shortcut = Conv1D(filters=filters, kernel_size=1, strides=strides, padding='same', kernel_initializer=tf.keras.initializers.RandomUniform(seed=seed))(shortcut)
#             if pooling_size > 1:
#                 shortcut = AveragePooling1D(pool_size=pooling_size)(shortcut)
                
#         # --- Merge ---
#         x = Add()([x, shortcut])
#         # Note: Standard ResNet adds then activates. 
#         # But cnn_core ends with Pool (no final activation). 
#         # Since 'x' (Main) is already activated (Selu->BN), and shortcut is linear, 
#         # adding them preserves the signal property roughly. 
#         # We optionally could add Activation('selu') here, but let's try strict identity addition first.
#         # x = ReLU()(x) 
        
#     output_layer = Flatten()(x) #shape of output layer is: (None, filters * reduced_length)
#     return output_layer

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
        # x = ReLU()(x)  <-- Removed to match cnn_core's activation logic (integrated in Conv)
        
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
        # Note: Standard ResNet adds then activates. 
        # But cnn_core ends with Pool (no final activation). 
        # Since 'x' (Main) is already activated (Selu->BN), and shortcut is linear, 
        # adding them preserves the signal property roughly. 
        # We optionally could add Activation('selu') here, but let's try strict identity addition first.
        # x = ReLU()(x) 
        
    output_layer = Flatten()(x) #shape of output layer is: (None, filters * reduced_length)
    return output_layer

def resnet_core_L2(inputs_core, convolution_blocks, kernel_size, filters, strides, pooling_size, seed = 42, dropout_rate=0.3, regularization_factor=1e-4):
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
        x = Conv1D(kernel_size=(kernel_size[block],), kernel_regularizer=tf.keras.regularizers.l2(regularization_factor), strides=strides, filters=filters, activation='selu', padding='same', kernel_initializer=tf.keras.initializers.RandomUniform(seed=seed))(x)
        x = BatchNormalization()(x)
        # x = ReLU()(x)  <-- Removed to match cnn_core's activation logic (integrated in Conv)
        
        if pooling_size > 1:
            x = AveragePooling1D(pool_size=pooling_size)(x)
        
        if dropout_rate and dropout_rate > 0:
            x = SpatialDropout1D(dropout_rate, seed=seed)(x)
            
        # --- Shortcut Branch ---
        if (strides > 1) or (pooling_size > 1) or (shortcut.shape[-1] != filters):
            # Projection: Use kernel_size=1. 
            # Note: We do NOT apply SELU here usually, but since the main path is activated, 
            # we simply project linearly to match dimensions for the addition.
            shortcut = Conv1D(filters=filters, kernel_size=1, strides=strides, padding='same', kernel_initializer=tf.keras.initializers.RandomUniform(seed=seed), kernel_regularizer=tf.keras.regularizers.l2(regularization_factor))(shortcut)
            if pooling_size > 1:
                shortcut = AveragePooling1D(pool_size=pooling_size)(shortcut)
            if dropout_rate and dropout_rate > 0:
                shortcut = SpatialDropout1D(dropout_rate, seed=seed)(shortcut)
                
        # --- Merge ---
        x = Add()([x, shortcut])
        # Note: Standard ResNet adds then activates. 
        # But cnn_core ends with Pool (no final activation). 
        # Since 'x' (Main) is already activated (Selu->BN), and shortcut is linear, 
        # adding them preserves the signal property roughly. 
        # We optionally could add Activation('selu') here, but let's try strict identity addition first.
        # x = ReLU()(x) 
        
    output_layer = Flatten()(x) #shape of output layer is: (None, filters * reduced_length)
    return output_layer

# def cnn_core(inputs_core,convolution_blocks , kernel_size,filters, strides , pooling_size,seed = 42):
#     x = inputs_core
#     for block in range(convolution_blocks):
#         x = Conv1D(kernel_size=(kernel_size[block],), strides=strides, filters=filters, activation='selu', padding='same',kernel_initializer=tf.keras.initializers.RandomUniform(seed=seed))(x)    
#         x = BatchNormalization()(x)
#         x = AveragePooling1D(pool_size = pooling_size)(x)
        
#     output_layer = Flatten()(x) 

#     return output_layer

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
        x = Dense(dense_units, activation='selu',kernel_initializer=tf.keras.initializers.RandomUniform(seed=seed))(x)
    if activated:
        output_layer = Dense(256,activation ='softmax' ,kernel_initializer=tf.keras.initializers.RandomUniform(seed=seed))(x)  
    else:
        output_layer = Dense(256,kernel_initializer=tf.keras.initializers.RandomUniform(seed=seed))(x)   
    return output_layer    

def dense_core_l2(inputs_core,dense_blocks,dense_units,activated = False,seed = 42,regularization_factor = 0.01):
    x = inputs_core    
    for block in range(dense_blocks):
        x = Dense(dense_units, activation='selu',kernel_initializer=tf.keras.initializers.RandomUniform(seed=seed),kernel_regularizer=tf.keras.regularizers.l2(regularization_factor))(x)
    if activated:
        output_layer = Dense(256,activation ='softmax' ,kernel_initializer=tf.keras.initializers.RandomUniform(seed=seed),kernel_regularizer=tf.keras.regularizers.l2(regularization_factor))(x)  
    else:
        output_layer = Dense(256,kernel_initializer=tf.keras.initializers.RandomUniform(seed=seed),kernel_regularizer=tf.keras.regularizers.l2(regularization_factor))(x)   
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
        # non_shared_branch.append(tf.expand_dims(x, 2))
        x_reshaped = Reshape((x.shape[1],1))(x)
        # non_shared_branch.append(x_reshaped)
        # Lambda(lambda t: tf.expand_dims(t, 2))(x)
        # x_reshaped = Lambda(lambda t: tf.expand_dims(t, 2))(x)
        non_shared_branch.append(x_reshaped)
    x = Concatenate(axis = 2)(non_shared_branch)
   
    for block in range(shared_block):
        x = SharedWeightsDenseLayer(input_dim = x.shape[1],units = units,shares = 14,seed = seed)(x)        
    output_layer = SharedWeightsDenseLayer(input_dim = x.shape[1],units = 256,activation = False,shares = 14,seed = seed)(x)   
    return output_layer 

# set dropout to 0.1–0.3 to enable dropout
def dense_core_shared_L2(inputs_core, shared_block=1, non_shared_block=1, units=64, branches=14, seed=42, dropout_rate=0.3, regularization_factor=1e-4):
    non_shared_branch = []
    for _ in range(branches):
        x = inputs_core
        for _ in range(non_shared_block):
            x = Dense(units, activation='selu', kernel_initializer=tf.keras.initializers.RandomUniform(seed=seed), kernel_regularizer=l2(regularization_factor))(x)
            if dropout_rate > 0:
                x = Dropout(dropout_rate, seed=seed)(x)

        # reshape to (batch, features, 1) for concat on axis=2
        x_reshaped = Reshape((x.shape[1], 1))(x)
        non_shared_branch.append(x_reshaped)

    # concat branches on the “share” axis -> shape (batch, features, branches)
    x = Concatenate(axis=2)(non_shared_branch)

    # shared blocks over branches
    for _ in range(shared_block):
        x = SharedWeightsDenseLayer(input_dim=x.shape[1], units=units, shares=branches, seed=seed)(x)
        if dropout_rate > 0:
            x = Dropout(dropout_rate, seed=seed)(x)

    output_layer = SharedWeightsDenseLayer(input_dim=x.shape[1], units=256, activation=False, shares=branches, seed=seed)(x)
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

#### Training high level function #####
def train_model(training_type,byte,seed,model_name=None, epochs = 100, resnet = False, learning_rate=0.001, batch_size=250, convolution_blocks=1, pooling_size=2, filters=4, strides=17, dropout_rate=0.3, regularization_factor=1e-4):
    seed = int(seed)
    epochs = epochs
    batch_size = batch_size
    n_traces = 50000
    single_task = False
    convolution_blocks = convolution_blocks

    # Print training summary (GeneralArch style)
    print(f"=== Training {training_type} | seed={seed} ===")
    print(f"Bytes: 2-15 | ResNet: {resnet}")
    print(f"Traces: {n_traces} | Epochs: {epochs}")
    print(f"LR: {learning_rate} (static)")
    reg_parts = []
    if regularization_factor > 0:
        reg_parts.append(f"L2={regularization_factor}")
    if dropout_rate > 0:
        reg_parts.append(f"dropout={dropout_rate}")
    if reg_parts:
        print(f"Regularization: {', '.join(reg_parts)}")

    if 'single_task' in training_type:
        single_task = True
        X_profiling , validation_data = load_dataset(byte,target = 't1' if 'subin' in training_type else 's1',n_traces = n_traces,dataset = 'training')
        model_t = 'model_{}'.format(training_type) 
    elif ('multi_task_single_target_one_shared_mask' in training_type) or ('multi_task_single_target_multi_shares' in training_type) :
        X_profiling , validation_data = load_dataset_multi('t1',n_traces = n_traces,dataset = 'training') 
        model_t = 'model_{}'.format(training_type)     
        
    elif 'multi_task_single_target' in training_type:
        X_profiling , validation_data = load_dataset_multi('s1',n_traces = n_traces,dataset = 'training') 
        model_t = 'model_{}'.format(training_type)

    elif 'multi_task_affine' in training_type:
        # Load dataset with all affine components
        # We need alpha, beta, etc. which load_dataset_multi should now handle if they exist
        # We target 's1' usually for the share components
        X_profiling , validation_data = load_dataset_multi('s1', n_traces = n_traces, dataset = 'training')
        model_t = 'model_{}'.format(training_type)

    elif 'multi_task_shared_branch_transformer' in training_type:
        X_profiling , validation_data = load_dataset_multi('t1',n_traces = n_traces,dataset = 'training') 
        model_t = 'model_{}'.format(training_type)

    elif 'multi_task_single_target_one_shared_mask_shared_branch_no_xor' in training_type:
        X_profiling , validation_data = load_dataset_multi('s1',n_traces = n_traces,dataset = 'training') 
        model_t = 'model_{}'.format(training_type)

    elif 'multi_task_single_target_one_shared_mask_shared_branch_Transformer_branch' in training_type:
        X_profiling , validation_data = load_dataset_multi('s1',n_traces = n_traces,dataset = 'training') 
        model_t = 'model_{}'.format(training_type)

    elif 'multi_task_single_target_one_shared_mask_shared_branch_general_masking' in training_type:
        X_profiling , validation_data = load_dataset_multi('t1',n_traces = n_traces,dataset = 'training') 
        model_t = 'model_{}'.format(training_type)

    # elif 'multi_task_shared_branch_transformer' in training_type:
    #     model = model_multi_task_shared_branch_transformer(
    #         input_length=window,
    #         seed=seed,
    #         resnet=resnet,
    #         learning_rate=learning_rate,
    #         convolution_blocks=convolution_blocks
    #     )
    #     single_task = False
        
    else:
        X_profiling , validation_data = load_dataset_multi('t1',n_traces = n_traces,dataset = 'training') 
        model_t = 'model_multi_task'
    
    window =  X_profiling.element_spec[0]['traces'].shape[0]
    monitor = 'val_accuracy'
    mode = 'max'

    # m_s
    if single_task:
        
        model = model_single_task(input_length = window, resnet = resnet, convolution_blocks=convolution_blocks, learning_rate=learning_rate)       
   
    # m_d  SBOX OUTPUT
    elif model_t == 'model_multi_task_single_target':
        model = model_multi_task_single_target(input_length = window,seed = seed, resnet = resnet, convolution_blocks=convolution_blocks, learning_rate=learning_rate)         
        monitor = 'val_loss'   
        mode = 'min'    
    # m_{nt * d}   SBOX OUTPUT
    elif model_t == 'model_multi_task_single_target_not_shared':

        model = model_multi_task_single_target_not_shared(input_length = window,seed = seed, resnet = resnet, convolution_blocks=convolution_blocks, learning_rate=learning_rate)         
        monitor = 'val_loss'   
        mode = 'min'     
    # m_{nt + d - 1}  SBOX INPUT 
    elif model_t == 'model_multi_task_single_target_one_shared_mask':
        
        model = model_multi_task_single_target_one_shared_mask(input_length = window, resnet = resnet, convolution_blocks=convolution_blocks, learning_rate=learning_rate)                  
        monitor = 'val_loss'   
        mode = 'min'
    # m_d  SBOX INPUT
    elif model_t == 'model_multi_task_single_target_one_shared_mask_shared_branch':
        
        model = model_multi_task_single_target_one_shared_mask_shared_branch(input_length = window,seed = seed, resnet = resnet, learning_rate=learning_rate, convolution_blocks=convolution_blocks, pooling_size=pooling_size, filters=filters, strides=strides)                  
        monitor = 'val_loss'   
        mode = 'min'             
    
    # Affine Masking (ASCADv2)
    elif 'multi_task_affine' in model_t:
        shared_branch = 'shared' in model_t
        model = model_multi_task_affine(input_length = window, seed = seed, resnet = resnet, learning_rate=learning_rate, shared_branch=shared_branch, convolution_blocks=convolution_blocks, pooling_size=pooling_size, filters=filters, strides=strides)
        monitor = 'val_loss'
        mode = 'min'
    
    # No XOR layer
    elif model_t == 'model_multi_task_single_target_one_shared_mask_shared_branch_no_xor':
        model = model_multi_task_single_target_one_shared_mask_shared_branch_no_xor(input_length = window,seed = seed, resnet = resnet, learning_rate=learning_rate, convolution_blocks=convolution_blocks, pooling_size=pooling_size, filters=filters, strides=strides, dropout_rate=dropout_rate, regularization_factor=regularization_factor)                  
        monitor = 'val_loss'   
        mode = 'min'  
    
    # Transformer with shared branch
    elif model_t == 'model_multi_task_shared_branch_transformer':
        model = model_multi_task_shared_branch_transformer(
            input_length=window,
            seed=seed,
            resnet=resnet,
            learning_rate=learning_rate,
            convolution_blocks=convolution_blocks,
            pooling_size=pooling_size,
            filters=filters,
            strides=strides,
            dropout=dropout_rate
        )
        monitor = 'val_loss'
        mode = 'min'

    elif model_t == 'model_multi_task_single_target_one_shared_mask_shared_branch_Transformer_branch':
        model = model_multi_task_single_target_one_shared_mask_shared_branch_Transformer_branch(
            input_length=window,
            seed=seed,
            resnet=resnet,
            learning_rate=learning_rate,
            convolution_blocks=convolution_blocks,
            pooling_size=pooling_size,
            filters=filters,
            strides=strides,
            # dropout_rate=dropout_rate,
            # regularization_factor=regularization_factor
        )
        monitor = 'val_loss'
        mode = 'min'

    elif model_t == 'model_multi_task_single_target_one_shared_mask_shared_branch_general_masking':
        model = model_multi_task_single_target_one_shared_mask_shared_branch_general_masking(
            input_length=window,
            seed=seed,
            resnet=resnet,
            learning_rate=learning_rate,
            convolution_blocks=convolution_blocks,
            pooling_size=pooling_size,
            filters=filters,
            strides=strides
        )
        monitor = 'val_loss'
        mode = 'min'
          
    else:
        print('Some error here')

    
    X_profiling = X_profiling.shuffle(len(X_profiling)).batch(batch_size) 
    validation_data = validation_data.batch(batch_size)
    # file_name = '{}_{}'.format(model_t,byte) 
    file_name = '{}_{}'.format(model_name,byte) if model_name else '{}_{}'.format(model_t,byte)
    
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
    parser.add_argument("--model_name", type=str, default='model_multi_task_single_target_one_shared_mask', help="Name of the model to be trained")
    # parser.add_argument("--byte", type=int, default=1, help="Byte to be trained")
    parser.add_argument("--seed", type=int, default=None, help="Seed to be used")
    parser.add_argument("--training_type", type=str, default='multi_task_single_target_one_shared_mask', help="Training type")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    # parser.add_argument("--resnet", type=bool, default=False, help="Use ResNet")
    parser.add_argument("--resnet", action='store_true', default=False, help="Use ResNet")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=250, help="Batch size")
    parser.add_argument("--convolution_blocks", type=int, default=1, help="Number of convolution blocks")
    parser.add_argument("--pooling_size", type=int, default=2, help="Pooling size")
    parser.add_argument("--filters", type=int, default=4, help="Number of filters")
    parser.add_argument("--strides", type=int, default=17, help="Strides")
    parser.add_argument("--dropout_rate", type=float, default=0.3, help="Dropout rate for L2 models")
    parser.add_argument("--regularization_factor", type=float, default=1e-4, help="L2 regularization factor for L2 models")

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
     
    TARGETS = {}
    
    # training_types = ['single_task_subout','single_task_subin']
    # training_types = ['multi_task_single_target_one_shared_mask']
    training_types = [training_type_arg]



    seeds_random = np.random.randint(0,9999,size = 9)
    # Because 42
    # seeds_random = np.concatenate([[42],seeds_random],axis = 0)
    seeds_random = [seed] if seed else np.concatenate([[42],seeds_random],axis = 0)
    
    for seed in seeds_random:
        tf.random.set_seed(seed)
        np.random.seed(seed)
        for training_type in training_types:
            
            # Depending on your setup, you might need to remove the "Process"
            
            if not 'single_task' in training_type:
                process_eval = Process(target=train_model, args=(training_type,'all',seed, model_name_arg, epochs, resnet, learning_rate, batch_size, convolution_blocks, pooling_size, filters, strides, dropout_rate, regularization_factor))
                process_eval.start()
                process_eval.join()      
            else:
                for byte in range(2,16):       
                    process_eval = Process(target=train_model, args=(training_type,byte,seed, model_name_arg, epochs, resnet, learning_rate, batch_size, convolution_blocks, pooling_size, filters, strides, dropout_rate, regularization_factor))
                    process_eval.start()
                    process_eval.join()       
                    
            # if not 'single_task' in training_type:
            #     train_model(training_type,'all',seed)

            # else:
            #     for byte in range(2,16):       
            #         train_model(training_type,byte,seed)
                    
                      

    print("$ Done !")
            
        
        
