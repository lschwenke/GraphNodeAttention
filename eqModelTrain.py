import os
from sacred import Experiment
import numpy as np
import seml
import numpy as np
import random
import warnings

import tensorflow as tf
from tensorflow.keras.callbacks import *
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.keras.layers import *


from modules import helper
from modules import gcnModels
from modules import LASA


from sklearn.model_selection import train_test_split
from pyts.approximation import SymbolicFourierApproximation
from pyts.approximation import SymbolicAggregateApproximation
from sklearn.metrics import mean_squared_error, mean_absolute_error


import datetime

# some helper methods:
def print_time():
    parser = datetime.datetime.now() 
    return parser.strftime("%d-%m-%Y %H:%M:%S")  

def normalize(inputs): 
    normalized = []
    for eq in inputs:
        maks = np.max(np.abs(eq))
        if maks != 0:
            normalized.append(eq/maks)
        else:
            normalized.append(eq)
    return np.array(normalized)      

def targets_to_list(targets): 
    targets = targets.transpose(2,0,1)

    targetList = []
    for i in range(0, len(targets)):
        targetList.append(targets[i,:,:])
        
    return targetList



def k_fold_split(inputs, targets, seed): 

    # make sure everything is seeded
    import os
    os.environ['PYTHONHASHSEED']=str(seed)
    import random
    random.seed(seed)
    np.random.seed(seed)
    np.random.permutation(seed)
    tf.random.set_seed(seed)
    
    p = np.random.permutation(len(targets))
    
    print('min of p = ',np.array(p)[50:100].min())
    print('max of p = ',np.array(p)[50:100].max())
    print('mean of p = ',np.array(p)[50:100].mean())
    inputs = inputs[p]
    targets = targets[p]

    
    ind = int(len(inputs)/5)
    inputsK = []
    targetsK = []

    for i in range(0,5-1):
        inputsK.append(inputs[i*ind:(i+1)*ind])
        targetsK.append(targets[i*ind:(i+1)*ind])

    
    inputsK.append(inputs[(i+1)*ind:])
    targetsK.append(targets[(i+1)*ind:])
  
    
    return inputsK, targetsK
        
def merge_splits(inputs, targets, k): # houden
    if k != 0:
        z=0
        inputsTrain = inputs[z]
        targetsTrain = targets[z]
    else:
        z=1
        inputsTrain = inputs[z]
        targetsTrain = targets[z]

    for i in range(z+1, 5):
        if i != k:
            inputsTrain = np.concatenate((inputsTrain, inputs[i]))
            targetsTrain = np.concatenate((targetsTrain, targets[i]))
    
    return inputsTrain, targetsTrain, inputs[k], targets[k]


### Experiment code 
ex = Experiment()
seml.setup_logger(ex)


@ex.post_run_hook
def collect_stats(_run):
    seml.collect_exp_stats(_run)


@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(seml.create_mongodb_observer(db_collection, overwrite=overwrite))


class ExperimentWrapper:
    """
    A simple wrapper around a sacred experiment, making use of sacred's captured functions with prefixes.
    This allows a modular design of the configuration, where certain sub-dictionaries (e.g., "data") are parsed by
    specific method. This avoids having one large "main" function which takes all parameters as input.
    """

    def __init__(self, init_all=True):
        if init_all:
            self.init_all()

    #init before the experiment!
    @ex.capture(prefix="init")
    def baseInit(self, seed_value: int, patience: int, model_chosen):

        self.model_chosen = model_chosen #sys.argv[2]

        # set seeds
        self.seed= seed_value 
        self.seed_value = seed_value
        os.environ['PYTHONHASHSEED']=str(seed_value)
        random.seed(seed_value)
        tf.random.set_seed(seed_value)
        np.random.RandomState(seed_value)

        np.random.seed(seed_value)

        context.set_global_seed(seed_value)
        ops.get_default_graph().seed = seed_value

        #pip install tensorflow-determinism needed
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
        np.random.seed(seed_value)

        # create early stop
        self.es = tf.keras.callbacks.EarlyStopping(patience=patience, verbose=0, min_delta=0.001, monitor='val_loss', mode='min',baseline=None, restore_best_weights=True)
        

        #init gpu
        physical_devices = tf.config.list_physical_devices('GPU') 
        for gpu_instance in physical_devices: 
            tf.config.experimental.set_memory_growth(gpu_instance, True)


    # Load the dataset
    @ex.capture(prefix="data")
    def init_dataset(self, random_state_here: int, network_choice: str, nbins: int, doFT: bool, ncoef:int, iLen: int):
        self.network_choice = network_choice
        self.random_state_here = random_state_here
        self.nbins = nbins
        self.doFT = doFT
        self.ncoef = ncoef

        # create test and train data based on dataset and the given parameters
        if network_choice == 'network1':
            test_set_size = 0.2
            inputs = np.load('data/inputs_ci.npy', allow_pickle = True)
            targets = np.load('data/targets.npy', allow_pickle = True)
            
            graph_input = np.load('data/minmax_normalized_laplacian.npy', allow_pickle=True)
            graph_input = np.array([graph_input] * inputs.shape[0])

            graph_features = np.load('data/station_coords.npy', allow_pickle=True)
            graph_features = np.array([graph_features] * inputs.shape[0])

        if network_choice == 'network2':
            test_set_size = 0.2
            inputs = np.load('data/othernetwork/inputs_cw.npy', allow_pickle = True)
            targets = np.load('data/othernetwork/targets.npy', allow_pickle = True)
            
            graph_input = np.load('data/othernetwork/minmax_normalized_laplacian.npy', allow_pickle=True)
            
            graph_input = np.array([graph_input] * inputs.shape[0])

            graph_features = np.load('data/othernetwork/station_coords.npy', allow_pickle=True)
            graph_features = np.array([graph_features] * inputs.shape[0])

        self.train_inputs, self.test_inputs, self.traingraphinput , self.testgraphinput, self.train_graphfeature, self.test_graphfeature, self.train_targets, self.testTargets = train_test_split(inputs,graph_input, graph_features, targets, test_size=test_set_size, random_state=random_state_here)

        self.test_inputs = self.test_inputs[:, :, :iLen, :]
        self.train_inputs = self.train_inputs[:, :, :iLen, :]

        self.iLen = iLen 

        # SFA preprocessing and map data to [-1, 1] base on the relative position
        def trans(val, tDict) -> float:
            return tDict[val]
        if doFT:
            transformerSS = []
            sax = SymbolicAggregateApproximation(n_bins=nbins, strategy='uniform')

            vocab = sax._check_params(nbins)
            tDict = dict()
            for i in range(len(vocab)):

                halfSize = (len(vocab)-1)/2
                tDict[vocab[i]] =float((i - halfSize) / halfSize)

            X_new = []
            for i in range(self.train_inputs.shape[-1]):
                transformerS = SymbolicFourierApproximation(n_coefs=ncoef,n_bins=nbins, strategy='uniform')
                transformerSS.append(transformerS)
                transformerS.fit(self.train_inputs[:,:,:,i].reshape(-1,self.train_inputs.shape[-2]))
                t = transformerS.transform(self.train_inputs[:,:,:,i].reshape(-1,self.train_inputs.shape[-2]))

                t1 = t
                t11 = []
                for v in t1: 
                    t111 = []
                    for v2 in v:
                        t111.append(trans(v2, tDict))
                    t11.append(t111)
                tx =t11
                print(tx[0][0])

                X_new.append(np.expand_dims(tx, axis=2))
            X_new = np.concatenate(X_new, axis=2)
            X_new = X_new.reshape((self.train_inputs.shape[0],self.train_inputs.shape[1],ncoef, self.train_inputs.shape[-1]))
            X_new.shape
            self.train_inputs = X_new



            X_new = []
            for i in range(self.test_inputs.shape[-1]):
                t = transformerSS[i].transform(self.test_inputs[:,:,:,i].reshape(-1,self.test_inputs.shape[-2]))

                t2 = t
                t22 = []
                for v in t2: 
                    t222 = []
                    for v2 in v:
                        t222.append(trans(v2, tDict))
                    t22.append(t222)
                tx = t22
                print(tx[0][0])


                X_new.append(np.expand_dims(tx, axis=2))
            X_new1 = np.concatenate(X_new, axis=2)
            X_new1 = X_new1.reshape((self.test_inputs.shape[0],self.test_inputs.shape[1],ncoef, self.test_inputs.shape[-1]))
            X_new1.shape
            self.testInputs = X_new1
        else:
            self.testInputs =  normalize(self.test_inputs)
        self.inputsK, self.targetsK = k_fold_split(self.train_inputs, self.train_targets, self.seed)


    #all inits
    def init_all(self):
        """
        Sequentially run the sub-initializers of the experiment.
        """
        self.baseInit()
        self.init_dataset()


    def modelTrainChain(self, combi_list_s, mse_list, rmse_list, mae_list, k, tempName, modelN, doAbstract, doStations, localAvgThresholds, layerCombis, steps, useEmbed,firstRun,useSaves,epochs, header,numOfAttentionLayers,dffFaktor,transDropout,lastDropout,gcnNodes,finalLayer,filters,cnnKernal,trainInputs,train_graphinput,train_graphfeatureinput,trainTargets,valInputs,val_graphinput,val_graphfeatureinput,valTargets,testInputs,doPosEnc=False, batches=20, doMixedLasa=False): 

            
            # reshape the data as needed and create the model based on the data
            if modelN == 'transformer':
                if doStations:
                    print('newShapes')
                    print(trainInputs.shape)
                    valInputs = np.transpose(valInputs, [0,1,3,2])
                    trainInputs = np.transpose(trainInputs, [0,1,3,2])

                    if firstRun:
                        firstRun = False
                        testInputs = np.transpose(testInputs, [0,1,3,2])
                        testInputs = np.reshape(testInputs, (testInputs.shape[0],testInputs.shape[1], testInputs.shape[2] * testInputs.shape[3]))

                    print(trainInputs.shape)
                    valInputs = np.reshape(valInputs, (valInputs.shape[0],valInputs.shape[1], valInputs.shape[2] * valInputs.shape[3]))
                    trainInputs = np.reshape(trainInputs, (trainInputs.shape[0],trainInputs.shape[1], trainInputs.shape[2] * trainInputs.shape[3]))
                    print(trainInputs.shape)
                    model = gcnModels.buildTransModel(valInputs, doStations, self.seed, useEmbed, self.model_chosen, header, numOfAttentionLayers, dffFaktor, transDropout, lastDropout, gcnNodes, finalLayer, doPosEnc=doPosEnc)
                else:
                    print('newShapes')
                    print(trainInputs.shape)
                    valInputs = np.transpose(valInputs, [0,2,3,1])
                    trainInputs = np.transpose(trainInputs, [0,2,3,1])

                    if firstRun:
                        firstRun = False
                        testInputs = np.transpose(testInputs, [0,2,3,1])
                        testInputs = np.reshape(testInputs, (testInputs.shape[0],testInputs.shape[1], testInputs.shape[2] * testInputs.shape[3]))

                    print(trainInputs.shape)
                    valInputs = np.reshape(valInputs, (valInputs.shape[0],valInputs.shape[1], valInputs.shape[2] * valInputs.shape[3]))
                    trainInputs = np.reshape(trainInputs, (trainInputs.shape[0],trainInputs.shape[1], trainInputs.shape[2] * trainInputs.shape[3]))
                    print(trainInputs.shape)
                    model = gcnModels.buildTransModel(valInputs, doStations, self.seed, useEmbed, self.model_chosen, header, numOfAttentionLayers, dffFaktor, transDropout, lastDropout, gcnNodes, finalLayer, doPosEnc=False)
            elif modelN == 'old':
                model = gcnModels.build_old_model(valInputs[0].shape, self.seed, filters=filters, kernal_size=cnnKernal, finalDense=finalLayer)
            elif modelN == 'bloem':
                model = gcnModels.build_bloem_model(valInputs[0].shape, self.seed, filters=filters, kernal_size=cnnKernal, GCNNodes=gcnNodes, finalDense=finalLayer)

            # create checkpoint and train or load model!
            iteration_checkpoint = tf.keras.callbacks.ModelCheckpoint(
                tempName,
                monitor='val_loss',
                verbose=0,
                save_best_only=True
            )
            print(tempName)
            print(model.summary())

            if (os.path.isfile('saves/test'+str(k)+'.index') and useSaves):
                print('found weights to load! Won\'t train model!')
                model.load_weights('saves/test'+str(k))
            else:
                history = model.fit(x=[trainInputs,train_graphinput,train_graphfeatureinput], 
                                    y=targets_to_list(trainTargets),
                        epochs=epochs, batch_size=batches,
                        validation_data=([valInputs,val_graphinput,val_graphfeatureinput], targets_to_list(valTargets)),verbose=1,callbacks=[self.es,iteration_checkpoint])#
            
            # predict model on test data and calculate the scores
            print()
            print('Fold number:' + str(k))
            predictions = model.predict([testInputs,self.testgraphinput, self.test_graphfeature])

            new_predictions = np.array(predictions)
            new_predictions = np.swapaxes(new_predictions,0,2)
            new_predictions = np.swapaxes(new_predictions,0,1)
            
            MSE = []
            for i in range(0,5):
                MSE.append(mean_squared_error(self.testTargets[:,:,i], new_predictions[:,:,i]))
            print('mse = ',np.array(MSE).mean())
            MSE = np.array(MSE).mean()
            
            RMSE = []
            for i in range(0,5):
                RMSE.append(mean_squared_error(self.testTargets[:,:,i], new_predictions[:,:,i], squared=False))
            print('rmse = ',np.array(RMSE).mean())
            RMSE = np.array(RMSE).mean()
            
            MAE = []
            for i in range(0,5):
                MAE.append(mean_absolute_error(self.testTargets[:,:,i], new_predictions[:,:,i]))
            print('MAE = ',np.array(MAE).mean())
            MAE = np.array(MAE).mean()
            

            mse_list.append(MSE)
            rmse_list.append(RMSE)
            mae_list.append(MAE)

            tf.keras.backend.clear_session()

            # do lasa
            if doAbstract:

                earlyPredictorZ = tf.keras.Model(model.inputs, model.layers[2].output)
                doMax = False
                takeAvg = True

                # threshold and step loop with each combi
                for thresholdSet in localAvgThresholds:
                    for order in layerCombis:
                        for step1 in steps:
                            for step2 in steps:
                                for step3 in steps:


                                    k = order+step1+step2+step3+str(thresholdSet[0])+','+str(thresholdSet[1])

                                    if(k not in combi_list_s.keys()):
                                        d = dict()
                                        d['mse'] = []
                                        d['rmse'] = []
                                        d['mae'] = []
                                        d['reduction'] = []
                                        if doMixedLasa:
                                            d['mse lasa seq'] = []
                                            d['rmse lasa seq'] = []
                                            d['mae lasa seq'] = []
                                            d['double lasa'] = dict()
                                        combi_list_s[k] = d
                                    else:
                                        d = combi_list_s[k]
                
                                    # new abstracted data
                                    newTrain, trainReduction, skipCounterTrain = LASA.abstractDataS([trainInputs,train_graphinput,train_graphfeatureinput] , earlyPredictorZ, order, step1, step2, step3, doMax, thresholdSet, takeAvg = takeAvg, heatLayer = 0, interpolate=False, useEmbed = False, doFidelity=False)
                                    newVal, valReduction, skipCounterVal = LASA.abstractDataS([valInputs,val_graphinput,val_graphfeatureinput], earlyPredictorZ, order, step1, step2, step3, doMax, thresholdSet, takeAvg = takeAvg, heatLayer = 0, interpolate=False, useEmbed = False, doFidelity=False)
                                    newTest, testReduction, skipCounterTest = LASA.abstractDataS([testInputs,self.testgraphinput, self.test_graphfeature] , earlyPredictorZ, order, step1, step2, step3, doMax, thresholdSet, takeAvg = takeAvg, heatLayer = 0, interpolate=False, useEmbed = False, doFidelity=False)

                                    # build new model for test and train it and evaluate it
                                    model2 = gcnModels.buildTransModel(newVal, doStations, self.seed, useEmbed, self.model_chosen, header, numOfAttentionLayers, dffFaktor, transDropout, lastDropout, gcnNodes, finalLayer, doPosEnc=False)

                                    tempName2 = tempName + k +'.h5'
                                    iteration_checkpoint = tf.keras.callbacks.ModelCheckpoint(
                                        tempName2,
                                        monitor='val_loss',
                                        verbose=0,
                                        save_best_only=True
                                    )

                                    print(model2.summary())

                                    history = model2.fit(x=[newTrain,train_graphinput,train_graphfeatureinput], 
                                                        y=targets_to_list(trainTargets),
                                            epochs=epochs, batch_size=20,
                                            validation_data=([newVal,val_graphinput,val_graphfeatureinput], targets_to_list(valTargets)),verbose=1,callbacks=[self.es,iteration_checkpoint])#

                                    print()
                                    print('total number of epochs ran = ',len(history.history['loss']))
                                    print('Fold number:' + str(k))
                                    predictions = model2.predict([newTest,self.testgraphinput, self.test_graphfeature])

                                    new_predictions = np.array(predictions)
                                    new_predictions = np.swapaxes(new_predictions,0,2)
                                    new_predictions = np.swapaxes(new_predictions,0,1)

                                    MSE = []
                                    for i in range(0,5):
                                        MSE.append(mean_squared_error(self.testTargets[:,:,i], new_predictions[:,:,i]))
                                    print('mse = ',np.array(MSE).mean())
                                    MSE = np.array(MSE).mean()

                                    RMSE = []
                                    for i in range(0,5):
                                        RMSE.append(mean_squared_error(self.testTargets[:,:,i], new_predictions[:,:,i], squared=False))
                                    print('rmse = ',np.array(RMSE).mean())
                                    RMSE = np.array(RMSE).mean()

                                    MAE = []
                                    for i in range(0,5):
                                        MAE.append(mean_absolute_error(self.testTargets[:,:,i], new_predictions[:,:,i]))
                                    print('MAE = ',np.array(MAE).mean())
                                    MAE = np.array(MAE).mean()

                                    d['mse'].append(MSE)
                                    d['rmse'].append(RMSE)
                                    d['mae'].append(MAE)
                                    d['reduction'].append(np.average(testReduction))

                                    tf.keras.backend.clear_session()

                                    if doMixedLasa:
                                        nnewTrain = newTrain.reshape((newTrain.shape[0], newTrain.shape[1], 3, -1))
                                        nnewVal = newVal.reshape((newVal.shape[0], newVal.shape[1], 3, -1))
                                        nnewTest = newTest.reshape((newTest.shape[0], newTest.shape[1], 3, -1))

                                        if doStations:
                                            nnewVal = np.transpose(nnewVal, [0,1,3,2])
                                            nnewTrain = np.transpose(nnewTrain, [0,1,3,2])
                                            nnewTest = np.transpose(nnewTest, [0,1,3,2])
                                        else:
                                            nnewVal = np.transpose(nnewVal, [0,3,1,2])
                                            nnewTrain = np.transpose(nnewTrain, [0,3,1,2])
                                            nnewTest = np.transpose(nnewTest, [0,3,1,2])
                                        

                                        #nnewTest = np.reshape(nnewTest, (nnewTest.shape[0],nnewTest.shape[1], nnewTest.shape[2] * nnewTest.shape[3]))

                                        #print(nnewTrain.shape)
                                        #nnewVal = np.reshape(nnewVal, (nnewVal.shape[0],nnewVal.shape[1], nnewVal.shape[2] * nnewVal.shape[3]))
                                        #nnewTrain = np.reshape(nnewTrain, (nnewTrain.shape[0],nnewTrain.shape[1], nnewTrain.shape[2] * nnewTrain.shape[3]))
                                        #print(nnewTrain.shape)

                                        #todo check if new train values get used!
                                        self.modelTrainChain(d['double lasa'], d['mse lasa seq'], d['rmse lasa seq'], d['mae lasa seq'], k, tempName, modelN, doAbstract, (not doStations), localAvgThresholds, layerCombis, steps, useEmbed,True,useSaves,epochs, header,numOfAttentionLayers,dffFaktor,transDropout,lastDropout,gcnNodes,finalLayer,filters,cnnKernal, nnewTrain,train_graphinput,train_graphfeatureinput,trainTargets,nnewVal,val_graphinput,val_graphfeatureinput,valTargets,nnewTest, doPosEnc=False, batches=20, doMixedLasa=False)



    # One experiment run with a certain config set. MOST OF THE IMPORTANT STUFF IS DONE HERE!!!!!!!
    @ex.capture(prefix="model")
    def trainExperiment(self, useEmbed: bool, useSaves: bool, header: int, numOfAttentionLayers: int, dffFaktor: int, transDropout: int, lastDropout: int, gcnNodes: int, finalLayer: int, doStations: bool, doAbstract: bool, doMixedLasa: bool, cnnKernal: int, epochs: int, filters, localAvgThresholds, layerCombis, steps, modelN): #, foldModel: int):

        # do some basic checks and name setups
        print("GPU is", "available" if tf.config.list_physical_devices("GPU") else "NOT AVAILABLE")
        print_time()
        warnings.filterwarnings('ignore')
        if modelN == 'transformer':   
            saveName = f"./results/results.n{self.network_choice}.rs{self.random_state_here}.ft{self.doFT}.s{self.nbins}.co{self.ncoef}.h{header}.aL{numOfAttentionLayers}.df{dffFaktor}.td{transDropout}.ld{lastDropout}.gn{gcnNodes}.fl{finalLayer}.st{doStations}.il{self.iLen}"
        else:
            saveName = f"./results/results.m{modelN}.n{self.network_choice}.rs{self.random_state_here}.ft{self.doFT}.s{self.nbins}.co{self.ncoef}.h{header}.f1{filters[0]}.f2{filters[1]}.ks{cnnKernal}.gn{gcnNodes}.fl{finalLayer}.il{self.iLen}"

        fullResults = dict()

        if os.path.isfile(saveName + '.pkl'):
            fullResults["Error"] = "Already done: " + saveName
            print('Already Done ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print("Already done: " + saveName)
        
            return "Already done: " + saveName #fullResults

        mse_list = []
        rmse_list = []
        mae_list = []
        combi_list_s = dict()

        firstRun = True

        
        # 5 folds loop
        for k in range(0,5):
            # prepare data
            tf.keras.backend.clear_session()
            if modelN == 'transformer':
                tempName = f"./models/f{k}.n{self.network_choice}.rs{self.random_state_here}.ft{self.doFT}.s{self.nbins}.co{self.ncoef}.h{header}.aL{numOfAttentionLayers}.df{dffFaktor}.td{transDropout}.ld{lastDropout}.gn{gcnNodes}.fl{finalLayer}.st{doStations}.il{self.iLen}.h5"
            else:
                tempName = f"./models/m{modelN}.f{k}.n{self.network_choice}.rs{self.random_state_here}.ft{self.doFT}.s{self.nbins}.co{self.ncoef}.h{header}.f1{filters[0]}.f2{filters[1]}.ks{cnnKernal}.gn{gcnNodes}.fl{finalLayer}.il{self.iLen}.h5"

            trainInputsAll, trainTargets, valInputsAll, valTargets = merge_splits(self.inputsK, self.targetsK, k)

            train_graphinput = self.traingraphinput[0:trainInputsAll.shape[0],:,:]
            train_graphfeatureinput = self.train_graphfeature[0:trainInputsAll.shape[0],:,:]

            val_graphinput = self.traingraphinput[0:valInputsAll.shape[0],:,:]
            val_graphfeatureinput = self.train_graphfeature[0:valInputsAll.shape[0],:,:]

            testInputs = self.testInputs
            if self.doFT:
                trainInputs = trainInputsAll
                valInputs = valInputsAll
            else:
                trainInputs = normalize(trainInputsAll)
                valInputs = normalize(valInputsAll)        

            self.modelTrainChain(combi_list_s, mse_list, rmse_list, mae_list, k, tempName, modelN, doAbstract, doStations, localAvgThresholds, layerCombis, steps, useEmbed,firstRun,useSaves,epochs,header,numOfAttentionLayers,dffFaktor,transDropout,lastDropout,gcnNodes,finalLayer,filters,cnnKernal,trainInputs,train_graphinput,train_graphfeatureinput,trainTargets,valInputs,val_graphinput,val_graphfeatureinput,valTargets,testInputs,doPosEnc=False, batches=20, doMixedLasa=doMixedLasa)


        print('-')
        print('-')
        print('-')
        print('-')
        print('all averages = ')
        print('mse score = ',np.array(mse_list).mean())
        print('rmse score = ',np.array(rmse_list).mean())
        print('mae score = ',np.array(mae_list).mean())

        fullResults['mse'] = mse_list
        fullResults['rmse'] = rmse_list
        fullResults['mae'] = mae_list
        fullResults['lasa'] = combi_list_s

        print("Done done")
        
        # save results of this experiment run
        print(saveName)
        helper.save_obj(fullResults, str(saveName))
        

        print_time()

        return saveName


  

# We can call this command, e.g., from a Jupyter notebook with init_all=False to get an "empty" experiment wrapper,
# where we can then for instance load a pretrained model to inspect the performance.
@ex.command(unobserved=True)
def get_experiment(init_all=False):
    print('get_experiment')
    experiment = ExperimentWrapper(init_all=init_all)
    return experiment


# This function will be called by default. Note that we could in principle manually pass an experiment instance,
# e.g., obtained by loading a model from the database or by calling this from a Jupyter notebook.
@ex.automain
def train(experiment=None):
    if experiment is None:
        experiment = ExperimentWrapper()
    return experiment.trainExperiment()