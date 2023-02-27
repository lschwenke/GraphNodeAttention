import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from spektral.layers import GCNConv
from modules import transformer

# build state of the art model from https://github.com/StefanBloemheuvel/GCNTimeseriesRegression
def build_bloem_model(input_shape, seed, filters=[16,32], kernal_size=25, GCNNodes=32, finalDense=256): # houden

    reg_const = 0.0001
    activation_func = 'relu'

    wav_input = layers.Input(shape=input_shape, name='wav_input')
    graph_input = layers.Input(shape=(39,39), name='graph_input')
    graph_features = layers.Input(shape=(39,2), name='graph_features')

    conv1 = layers.Conv1D(filters=filters[0], kernel_size=kernal_size, strides=2,  activation=activation_func, kernel_regularizer=regularizers.l2(reg_const), name='conv1')(wav_input)
    conv1 = layers.Conv1D(filters=filters[1], kernel_size=kernal_size, strides=2,  activation=activation_func, kernel_regularizer=regularizers.l2(reg_const), name='conv2')(conv1)

    conv1_new = tf.keras.layers.Reshape((39,conv1.shape[2] * conv1.shape[3]))(conv1)    

    conv1_new = layers.concatenate(inputs=[conv1_new, graph_features], axis=2)

    conv1_new = GCNConv(GCNNodes, activation='relu', use_bias=False, kernel_regularizer=regularizers.l2(reg_const))([conv1_new, graph_input])
    conv1_new = GCNConv(GCNNodes, activation='tanh', use_bias=False, kernel_regularizer=regularizers.l2(reg_const))([conv1_new, graph_input])

    conv1_new = layers.Flatten()(conv1_new)
    conv1_new = layers.Dropout(0.4, seed=seed)(conv1_new)

    merged = layers.Dense(finalDense)(conv1_new)

    pga = layers.Dense(39)(merged)
    pgv = layers.Dense(39)(merged)
    sa03 = layers.Dense(39)(merged)
    sa10 = layers.Dense(39)(merged)
    sa30 = layers.Dense(39)(merged)
    
    final_model = models.Model(inputs=[wav_input, graph_input, graph_features], outputs=[pga, pgv, sa03, sa10, sa30]) #, pgv, sa03, sa10, sa30
    rmsprop = tf.keras.optimizers.RMSprop(learning_rate=0.0001, rho=0.9, epsilon=None, decay=0.)
    final_model.compile(optimizer=rmsprop, loss='mse')
    
    return final_model

# create base model from https://github.com/StefanBloemheuvel/GCNTimeseriesRegression
def build_old_model(input_shape, seed, filters=[16,32], finalDense=256, kernal_size=25):

    reg_const = 0.0001
    activation_func = 'relu'

    wav_input = layers.Input(shape=input_shape, name='wav_input')
    
    conv1 = layers.Conv2D(filters[0], (1, kernal_size), strides=(1, 2),  activation=activation_func, kernel_regularizer=regularizers.l2(reg_const))(wav_input)
    conv1 = layers.Conv2D(filters[1], (1, kernal_size), strides=(1, 2),  activation=activation_func, kernel_regularizer=regularizers.l2(reg_const))(conv1)
    conv1 = layers.Conv2D(filters[1], (39, 5), strides=(39, 5),  activation=activation_func, padding = 'same', kernel_regularizer=regularizers.l2(reg_const))(conv1)
    
    conv1 = layers.Flatten()(conv1)
    conv1 = layers.Dropout(0.4, seed=seed)(conv1)

    graph_input = layers.Input(shape=(39,39), name='graph_input')
    graph_features = layers.Input(shape=(39,2), name='graph_features')
    graph_features_flattened = layers.Flatten()(graph_features)


    merged = layers.concatenate(inputs=[conv1, graph_features_flattened])
    merged = layers.Dense(finalDense)(merged)

    pga = layers.Dense(39)(merged)
    pgv = layers.Dense(39)(merged)
    sa03 = layers.Dense(39)(merged)
    sa10 = layers.Dense(39)(merged)
    sa30 = layers.Dense(39)(merged)
    
    final_model = models.Model(inputs=[wav_input, graph_input, graph_features], outputs=[pga, pgv, sa03, sa10, sa30]) #, pgv, sa03, sa10, sa30
    # final_model = models.Model(inputs=[wav_input,graph_features], outputs=[pga, pgv, sa03, sa10, sa30]) #, pgv, sa03, sa10, sa30
    rmsprop = tf.keras.optimizers.RMSprop(learning_rate=0.0001, rho=0.9, epsilon=None, decay=0.)

    # final_model.compile(optimizer=rmsprop, loss='mse', metrics=['mse'])
    final_model.compile(optimizer=rmsprop, loss='mse')#, metrics=['mse'])
    
    return final_model

# create transformer based GCN
def buildTransModel(data, doStations, seed, useEmbed, model_chosen, header=6, numOfAttentionLayers=2, dffFaktor=0.5, dropout=0, lastDropout=0.4, gcnNodes=32, finalLayer=256, dim=3, doPosEnc=False, maxLen=5000):
    reg_const = 0.0001
    activation_func = 'relu'
    seq_len1 = data.shape[1]
    seed_value = seed

    sens1 = data.shape[2]
    input_shape1 = (seq_len1, sens1)
    left_input1 = layers.Input(shape=input_shape1, name='wav_input')
    graph_input = layers.Input(shape=(39, 39), name='graph_input')
    graph_features = layers.Input(shape=(39, 2), name='graph_features')

    encoded = left_input1
    input_vocab_size = 0

    header = header
    dmodel = sens1
    dff = int(sens1 * dffFaktor) 
    rate = dropout  

    # create transformer encoder layer
    maskLayer = tf.keras.layers.Masking(mask_value=-2)
    encInput = maskLayer(encoded)

    
    if(useEmbed):
        encClass1 = transformer.Encoder(numOfAttentionLayers, dmodel, header, dff, maxLen, rate=rate,
                                        input_vocab_size=input_vocab_size + 2, maxLen=maxLen, doEmbedding=useEmbed, seed_value=seed_value)
    else:
        encClass1 = transformer.Encoder(numOfAttentionLayers, dmodel, header, dff, 5000, doPosEnc=doPosEnc, rate=rate,
                                        input_vocab_size=input_vocab_size + 2, maxLen=seq_len1, seed_value=seed_value)
    enc1, attention, fullAttention = encClass1(encInput)
    print('new shape what?')
    print(enc1.shape)
    conv1 = enc1

    conv1_new = conv1
    if model_chosen == 'nofeatures':
        print('went for no features version')
    if model_chosen == 'main':
        if not doStations:
            print(conv1.shape)
            conv1_new = tf.keras.layers.Reshape((conv1.shape[1], 3, 39))(conv1_new)
            conv1_new = tf.transpose(conv1_new, perm=[0,3,2,1])
            conv1_new = tf.keras.layers.Reshape((39,-1))(conv1_new)
        print('went for features version')
        conv1_new = layers.concatenate(
            inputs=[conv1_new, graph_features], axis=2)

    conv1_new = GCNConv(gcnNodes, activation='relu', use_bias=False,
                        kernel_regularizer=regularizers.l2(reg_const))([conv1_new, graph_input])
    conv1_new = GCNConv(gcnNodes, activation='tanh', use_bias=False,
                        kernel_regularizer=regularizers.l2(reg_const))([conv1_new, graph_input])

    conv1_new = layers.Flatten()(conv1_new)
    conv1_new = layers.Dropout(lastDropout, seed=seed)(conv1_new)#0.4

    merged = layers.Dense(finalLayer)(conv1_new)#64

    pga = layers.Dense(39)(merged)
    pgv = layers.Dense(39)(merged)
    sa03 = layers.Dense(39)(merged)
    sa10 = layers.Dense(39)(merged)
    sa30 = layers.Dense(39)(merged)

    final_model = models.Model(inputs=[left_input1, graph_input, graph_features], outputs=[
                               pga, pgv, sa03, sa10, sa30])  # , pgv, sa03, sa10, sa30
    
    learning_rate = transformer.CustomSchedule(128)

    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.99, 
                                    epsilon=1e-9)
    
    final_model.compile(optimizer=optimizer,
                loss='mse',run_eagerly=False)

    return final_model