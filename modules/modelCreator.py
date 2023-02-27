import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Input
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.python.keras.layers.merge import concatenate
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from modules import helper
from modules import transformer
from modules import LASA
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from pyts.approximation import SymbolicAggregateApproximation
import antropy as ant
import math
import os

#create the transformer model with given information
def createModel(splits, x_train, x_val, x_test, batchSize, seed_value, num_of_classes, numOfAttentionLayers, dmodel, header, dff, rate = 0.0, useEmbed=False):    
        x_trains = np.dsplit(x_train, splits)
        print(np.array(x_trains).shape)

        x_trainsBatch = np.dsplit(x_train[:batchSize], splits)

        x_tests = np.dsplit(x_test, splits)
        x_vals = np.dsplit(x_val, splits)
        maxLen = len(x_trains[0][0])
        print(maxLen)

        if(useEmbed):
            for i in range(splits):
                x_trains[i]= np.array([[" ".join([item[0] for item in x])] for x in x_trains[i]])
                x_trainsBatch[i]= np.array([[" ".join([item[0] for item in x])] for x in x_trainsBatch[i]])
                x_tests[i]= np.array([[" ".join([item[0] for item in x])] for x in x_tests[i]])
                x_vals[i]= np.array([[" ".join([item[0] for item in x])] for x in x_vals[i]])

        print(np.array(x_trains).shape)
        flattenArray = []
        inputShapes = []
        encClasses = []
        for i in range(len(x_trains)):
            mask = Input(1)
            x_part = np.array(x_trains[i])
            print(np.array(x_part).shape)
        
            seq_len1 = x_part.shape[1]

            if(useEmbed):
                left_input1 = Input(shape=(1,), dtype=tf.string)
            else:
                sens1 = x_part.shape[2]
                input_shape1 = (seq_len1, sens1)
                left_input1 = Input(input_shape1)

            inputShapes.append(left_input1)

            if(useEmbed):
                encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(
                    max_tokens=50, output_sequence_length=maxLen)
                encoder.adapt(x_part)

                encoded = encoder(left_input1)
                print(encoder.get_vocabulary())
                input_vocab_size = len(np.array(encoder.get_vocabulary()))
            else:
                encoded = left_input1
                input_vocab_size = 0
                
            #create transformer encoder layer 
            if(useEmbed):
                #rate=0.38
                encClass1 = transformer.Encoder(numOfAttentionLayers, dmodel, header, dff, maxLen, rate=rate, input_vocab_size = input_vocab_size + 2, maxLen = maxLen, doEmbedding=useEmbed, seed_value=seed_value)
            else:
                encClass1 = transformer.Encoder(numOfAttentionLayers, dmodel, header, dff, 5000, rate=rate, input_vocab_size = input_vocab_size + 2, maxLen = maxLen, seed_value=seed_value)
                
            encClasses.append(encClass1)

            maskLayer = tf.keras.layers.Masking(mask_value=-2)
            encInput = maskLayer(encoded)
            enc1, attention, fullAttention = encClass1(encInput)
            flatten1 = Flatten()(enc1)
            flattenArray.append(flatten1)
        

        # Merge nets
        if splits == 1:
            merged = flattenArray[0]
        else:
            merged = concatenate(flattenArray)

        output = Dense(num_of_classes, activation = "sigmoid")(merged)
        
        # Create combined model
        wdcnnt_multi = Model(inputs=inputShapes,outputs=(output))
        print(wdcnnt_multi.summary())
        
        print(wdcnnt_multi.count_params())
        
        tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=seed_value)

        learning_rate = transformer.CustomSchedule(32)
        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.99, 
                                     epsilon=1e-9)
        
        wdcnnt_multi.compile(optimizer=optimizer,
                    loss='mean_squared_error',
                    metrics=['accuracy'], run_eagerly=False)
        
        print('done')
        
        return wdcnnt_multi, inputShapes, x_trains, x_tests, x_vals

# building saving name for model weights
def getWeightName(name, fold, symbols, layers, abstractionType, header, learning = True, resultsPath = 'presults', results=False, usedp=False, doHeaders=True):
    #baseName = "./saves/weights-" + str(data_path_train.split('/')[-1].split('.')[0]) + '-size' + str(seqSize) + '-threshold' + maxString + '-input' + abstractionString + '-fold' + str(fold) + '-bins' + str(n_bins)

    if usedp:
        if results:
            baseName = "./"+resultsPath+"/results-" +name +' -fold: '+str(fold)+' -symbols: '+str(symbols)+ ' -layers: '+str(layers)+' -abstractionType: '+abstractionType
        else: 
            baseName = "./saves/weights-" +name +' -fold: '+str(fold)+' -symbols: '+str(symbols)+ ' -layers: '+str(layers)+' -abstractionType: '+abstractionType
    else:
        if results:
            baseName = "./"+resultsPath+"/results-" +name +' -fold '+str(fold)+' -symbols '+str(symbols)+ ' -layers '+str(layers)+' -abstractionType '+abstractionType
        else: 
            baseName = "./saves/weights-" +name +' -fold '+str(fold)+' -symbols '+str(symbols)+ ' -layers '+str(layers)+' -abstractionType '+abstractionType
    
    if doHeaders:
        baseName = baseName + ' -headers ' + str(header)
    if learning:
        return baseName + '-learning.tf'
    else:
        return baseName + '.tf'
    



def calcComplexityMetrics(newVal, newTest):
    valShifts = []
    smallerValSet = []
    for val in newVal:
        shifts = -1
        smallerSet = 2
        lastVal = val[0][0]
        rise = -3
        timeSkip = 1
        for v in val[1:]:
            v = v[0]

            if v == -2:
                timeSkip += 1
            elif helper.truncate((v - lastVal) / timeSkip) != rise:
                shifts += 1
                smallerSet += 1
                rise = helper.truncate((v - lastVal) / timeSkip)
                lastVal = v
                timeSkip = 1
            else:
                lastVal = v
        valShifts.append(shifts)
        smallerValSet.append(smallerSet)

    #valShifts = np.average(valShifts)
    #smallerValSet = np.average(smallerValSet)


    testShifts = []
    smallerTestSet = []
    print(newTest.shape)
    for val in newTest:
        shifts = -1
        smallerSet = 2
        lastVal = val[0][0]
        rise = -3
        timeSkip = 1
        for v in val[1:]:
            v = v[0]
            if v == -2:
                timeSkip += 1
            elif round(v - lastVal / timeSkip, 1) != rise:
                shifts += 1
                smallerSet += 1
                rise = round(v - lastVal / timeSkip, 1)
                lastVal = v
                timeSkip = 1
            else:
                lastVal = v
        testShifts.append(shifts)
        smallerTestSet.append(smallerSet)

    #testShifts = np.average(testShifts)
    #smallerTestSet = np.average(smallerTestSet)

    permutationEntropyVal = []
    spectralEntropyVal = []
    decompositionEntropyVal = []
    approximateEntropyVal = []
    sampleEntropyVal = []
    complexityEstimationVal = []
    spectralEntropyTest = []
    permutationEntropyTest = []
    decompositionEntropyTest = []
    approximateEntropyTest = []
    sampleEntropyTest = []
    complexityEstimationTest = []
    
    print('calc entropies val')
    for x in newVal.squeeze():
        permutationEntropyVal.append(ant.perm_entropy(x, normalize=True))
        spectralEntropyVal.append(ant.spectral_entropy(x, sf=100, method='welch', normalize=True, nperseg = len(x)))
        decompositionEntropyVal.append(ant.svd_entropy(x, normalize=True))
        approximateEntropyVal.append(ant.app_entropy(x))
        sampleEntropyVal.append(ant.sample_entropy(x))
        complexityEstimationVal.append(helper.ce(x))

    print('calc entropies tests')
    for x in newTest.squeeze():
        spectralEntropyTest.append(ant.spectral_entropy(x, sf=100, method='welch', normalize=True, nperseg = len(x)))
        permutationEntropyTest.append(ant.perm_entropy(x, normalize=True))
        decompositionEntropyTest.append(ant.svd_entropy(x, normalize=True))
        approximateEntropyTest.append(ant.app_entropy(x))
        sampleEntropyTest.append(ant.sample_entropy(x))
        complexityEstimationTest.append(helper.ce(x))


    valLen = len(newVal[0])
    testLen = len(newTest[0])

    complexityVal = {'permutationsEntropy': permutationEntropyVal, 'spectralEntropy': spectralEntropyVal, 'svdEntropy': decompositionEntropyVal, 'approximateEntropy': approximateEntropyVal, 'sample entropy': sampleEntropyVal, 'CE': complexityEstimationVal}
    complexityTest = {'permutationsEntropy': permutationEntropyTest, 'spectralEntropy': spectralEntropyTest, 'svdEntropy': decompositionEntropyTest, 'approximateEntropy': approximateEntropyTest, 'sample entropy': sampleEntropyTest, 'CE':  complexityEstimationTest}

    return complexityVal, complexityTest, smallerValSet, smallerTestSet, valShifts, testShifts

# do training for the given model def
def doAbstractedTraining(trainD, valD, testD, y_train1, y_val, y_testy, BATCH, seed_value, num_of_classes, dataName, fold, symbolCount, num_epochs, numOfAttentionLayers, dmodel, header, dff, skipDebugSaves=False, rMA = None, useEmbed = False, 
    reductionInt = 3, earlystop = None, useSaves=False, abstractionType=None, thresholdSet=None, order=None, step1=None, step2=None, step3=None, 
    doMax=False, abstraction = 0, earlyPredictorZ = None, takeAvg = True, rate=0.0, heatLayer = 0, calcComplexity=True, doFidelity=False):
    if(order != None):
        combination = order +'-' +step1 +'-' +step2 +'-' +step3
    else:
        combination = None

    if abstraction == 2:
        newTrain, trainReduction, skipCounterTrain = LASA.abstractDataS(trainD, earlyPredictorZ, order, step1, step2, step3, doMax, thresholdSet, takeAvg = takeAvg, heatLayer = heatLayer,useEmbed = useEmbed, doFidelity=doFidelity)
        newVal, valReduction, skipCounterVal = LASA.abstractDataS(valD, earlyPredictorZ, order, step1, step2, step3, doMax, thresholdSet, takeAvg = takeAvg, heatLayer = heatLayer,useEmbed = useEmbed, doFidelity=doFidelity)
        newTest, testReduction, skipCounterTest = LASA.abstractDataS(testD, earlyPredictorZ, order, step1, step2, step3, doMax, thresholdSet, takeAvg = takeAvg, heatLayer = heatLayer,useEmbed = useEmbed, doFidelity=doFidelity)
    elif abstraction == 3:
        newTrain, trainReduction, skipCounterTrain = LASA.abstractDataS(trainD, earlyPredictorZ, order, step1, step2, step3, doMax, thresholdSet, takeAvg = takeAvg, heatLayer = heatLayer, interpolate=False, useEmbed = useEmbed, doFidelity=doFidelity)
        newVal, valReduction, skipCounterVal = LASA.abstractDataS(valD, earlyPredictorZ, order, step1, step2, step3, doMax, thresholdSet, takeAvg = takeAvg, heatLayer = heatLayer, interpolate=False, useEmbed = useEmbed, doFidelity=doFidelity)
        newTest, testReduction, skipCounterTest = LASA.abstractDataS(testD, earlyPredictorZ, order, step1, step2, step3, doMax, thresholdSet, takeAvg = takeAvg, heatLayer = heatLayer, interpolate=False, useEmbed = useEmbed, doFidelity=doFidelity)
    elif abstraction == 4:
        newTrain, trainReduction, skipCounterTrain = GIA.abstractDataMixGTM(trainD, rMA, reductionInt, thresholdSet, useEmbed = useEmbed, doFidelity=doFidelity)#abstractDataMix(trainD, rMA)
        newVal, valReduction, skipCounterVal = GIA.abstractDataMixGTM(valD, rMA, reductionInt, thresholdSet, useEmbed = useEmbed, doFidelity=doFidelity)#abstractDataMix(valD, rMA)
        newTest, testReduction, skipCounterTest = GIA.abstractDataMixGTM(testD, rMA, reductionInt, thresholdSet, useEmbed = useEmbed, doFidelity=doFidelity)#abstractDataMix(testD, rMA)
    else:
        newTrain = trainD
        newVal = valD
        newTest = testD
        trainReduction = 0
        valReduction = 0
        testReduction = 0
        skipCounterTrain = 0
        skipCounterVal = 0
        skipCounterTest = 0
        
    if calcComplexity:
        complexityVal, complexityTest, smallerValSet, smallerTestSet, valShifts, testShifts = calcComplexityMetrics(newVal, newTest)
    else: 
        complexityVal = []
        complexityTest = []
        smallerValSet = 0
        smallerTestSet = 0
        valShifts, testShifts = 0,0
        
    print('newTrain during abstract training:')
    print(newTrain.shape)
            
    n_model2, inputs2, x_trains2, x_tests2, x_vals2 = createModel(1, newTrain, newVal, newTest , BATCH, seed_value, num_of_classes, numOfAttentionLayers, dmodel, header, dff, rate=rate)
    weightsName = getWeightName(dataName, fold, symbolCount, numOfAttentionLayers, abstractionType, header, learning=True)
    saveBest2 = transformer.SaveBest(weightsName)
    print(np.array(x_trains2).shape)
    print(np.array(x_vals2).shape)

    x_trains_mask = x_trains2
    
    if (os.path.isfile(getWeightName(dataName, fold, symbolCount, numOfAttentionLayers, abstractionType, header, learning=False) + '.index') and useSaves):
        print('found weights to load! Won\'t train model!')
        n_model2.load_weights(getWeightName(dataName, fold, symbolCount, numOfAttentionLayers, abstractionType, header, learning=False))
    else:
        print('No weights found! Start training model!')
        n_model2.fit(x_trains_mask, y_train1, validation_data = (x_vals2, y_val) , epochs = num_epochs, batch_size = BATCH, verbose=1, callbacks =[earlystop, saveBest2], shuffle = True)
        print('Model fitted!!!')
        n_model2.load_weights(getWeightName(dataName, fold, symbolCount, numOfAttentionLayers, abstractionType, header, learning=True))
        print('Best model loaded!')
        n_model2.save_weights(getWeightName(dataName, fold, symbolCount, numOfAttentionLayers, abstractionType, header, learning=False), overwrite=True)
        
    #print([lay.name for lay in n_model.layers])
    print('Starting earlyPredictor creation')
    if(useEmbed):
        earlyPredictor2 = tf.keras.Model(n_model2.inputs, n_model2.layers[3].output)
    else:
        earlyPredictor2 = tf.keras.Model(n_model2.inputs, n_model2.layers[2].output)

    # Predictions on the validation set
    print('starting val prediction')
    predictions2 = n_model2.predict(x_vals2)
    
    #print(len(attentionQ))
    #print(attentionQ[1])
    print('############################')
    predictions2 = np.argmax(predictions2,axis=1)+1

    # Measure this fold's accuracy on validation set compared to actual labels
    y_compare = np.argmax(y_val, axis=1)+1
    val_score2 = metrics.accuracy_score(y_compare, predictions2)
    val_pre = metrics.precision_score(y_compare, predictions2, average='macro')
    val_rec = metrics.recall_score(y_compare, predictions2, average='macro')
    val_f1= metrics.f1_score(y_compare, predictions2, average='macro')

    print('Do max:' + str(doMax))
    print(f"validation fold score with input {abstractionType}(accuracy): {val_score2}")

    # Predictions on the test set
    limit = 500
    test_predictions_loop2 = []
    for bor in range(int(math.ceil(len(x_tests2[0])/limit))):
        test_predictions_loop2.extend(n_model2.predict([x_tests2[0][bor*limit:(bor+1)*limit]]))

    attentionQ2 = None
    if (not skipDebugSaves) or abstraction < 2:
        attentionQ0 = []
        attentionQ1 = []
        attentionQ2 = []

        for bor in range(int(math.ceil(len(x_trains2[0])/limit))):
            attOut = earlyPredictor2.predict([x_trains2[0][bor*limit:(bor+1)*limit]])
            attentionQ0.extend(attOut[0]) 
            attentionQ1.extend(attOut[1])

            if len(attentionQ2) == 0:
                attentionQ2 = attOut[2]
            else:
                for k in range(len(attentionQ2)):
                    attentionQ2[k] = np.append(attentionQ2[k], attOut[2][k], 0)


        attentionQ2 = [attentionQ0, attentionQ1, attentionQ2]
    #attentionQ2 = earlyPredictor2.predict(x_tests2)
    
    # Append actual labels of the test set to empty list
    # y_testyy = [y-1 for y in y_testy]

    #oos_test_y2.append(y_testyy)
    #oos_test_prob2.append(test_predictions_loop2)
    test_predictions_loop2 = np.argmax(test_predictions_loop2, axis=1)+1

    # Measure this fold's accuracy on test set compared to actual labels
    test_score2 = metrics.accuracy_score(y_testy, test_predictions_loop2)
    test_pre = metrics.precision_score(y_testy, test_predictions_loop2, average='macro')
    test_rec = metrics.recall_score(y_testy, test_predictions_loop2, average='macro')
    test_f1= metrics.f1_score(y_testy, test_predictions_loop2, average='macro')

    print(f"test fold score with input {abstractionType}-{doMax}(accuracy): {test_score2}")

    train_predictions= []
    for bor in range(int(math.ceil(len(x_trains2[0])/limit))):
        train_predictions.extend(n_model2.predict([x_trains2[0][bor*limit:(bor+1)*limit]]))
    train_predictions = np.argmax(train_predictions, axis=1)+1

    

    if skipDebugSaves and abstraction >= 2:
        return [val_score2, val_pre, val_rec, val_f1], [test_score2, test_pre, test_rec, test_f1], [train_predictions, predictions2], test_predictions_loop2, None, [complexityVal, complexityTest], None, None, None, None, smallerValSet, smallerTestSet, valShifts, testShifts, None, None, None, None, [valReduction, testReduction], [skipCounterTrain,skipCounterVal,skipCounterTest], y_val, y_train1
    else:
        return [val_score2, val_pre, val_rec, val_f1], [test_score2, test_pre, test_rec, test_f1], [train_predictions, predictions2], test_predictions_loop2, n_model2, [complexityVal, complexityTest], x_trains2, x_tests2, x_vals2, attentionQ2, smallerValSet, smallerTestSet, valShifts, testShifts, earlyPredictor2, newTrain, newVal, newTest, [valReduction, testReduction], [skipCounterTrain,skipCounterVal,skipCounterTest], y_val, y_train1





def preprocessData(x_train1, x_val, X_test, y_train1, y_val, y_test, y_trainy, y_testy, binNr, symbolsCount, dataName, useEmbed = False, useSaves = False, doSymbolify = True, multiVariant=False):    
    
    x_test = X_test.copy()
    
    if(useEmbed):
        processedDataName = "./saves/"+str(dataName)+ '-bin' + str(binNr) + '-symbols' + str(symbolsCount) + '+embedding'
    else:
        processedDataName = "./saves/"+str(dataName)+ '-bin' + str(binNr) + '-symbols' + str(symbolsCount)
    fileExists = os.path.isfile(processedDataName +'.pkl')

    if(fileExists and useSaves):
        print('found file! Start loading file!')
        res = helper.load_obj(processedDataName)


        for index, v in np.ndenumerate(res):
            print(index)
            res = v
        res.keys()

        x_train1 = res['X_train']
        #x_train1 = res['X_val']
        x_test = res['X_test']
        x_val = res['X_val']
        X_train_ori = res['X_train_ori']
        X_test_ori = res['X_test_ori']
        y_trainy = res['y_trainy']
        y_train1 = res['y_train']
        y_test = res['y_test']
        y_testy = res['y_testy']
        y_val = res['y_val']
        X_val_ori = res['X_val_ori']
        print(x_test.shape)
        print(x_train1.shape)
        print(y_test.shape)
        print(y_train1.shape)
        print('SHAPES loaded')
        
    else:
        print('SHAPES:')
        print(x_test.shape)
        print(x_train1.shape)
        print(x_val.shape)
        print(y_test.shape)
        print(y_train1.shape)

        x_train1 = x_train1.squeeze()
        x_val = x_val.squeeze()
        x_test = x_test.squeeze()
        
        trainShape = x_train1.shape
        valShape = x_val.shape
        testShape = x_test.shape
        
        if multiVariant:
            X_test_ori = x_test.copy()
            X_val_ori = x_val.copy()
            X_train_ori = x_train1.copy()
            for i in range(trainShape[-1]):
                x_train2 = x_train1[:,:,i]
                x_val2 = x_val[:,:,i]
                x_test2 = x_test[:,:,i]
                print('####')
                print(x_train2.shape)

                trainShape2 = x_train2.shape
                valShape2 = x_val2.shape
                testShape2 = x_test2.shape
        
                scaler = StandardScaler()    
                scaler = scaler.fit(x_train1.reshape((-1,1)))
                x_train2 = scaler.transform(x_train2.reshape(-1, 1)).reshape(trainShape2)##
                x_val2 = scaler.transform(x_val2.reshape(-1, 1)).reshape(valShape2)
                x_test2 = scaler.transform(x_test2.reshape(-1, 1)).reshape(testShape2)

                sax = SymbolicAggregateApproximation(n_bins=symbolsCount, strategy='uniform')
                sax.fit(x_train2)

                if(useEmbed):
                    x_train2 = helper.symbolize(x_train2, sax)
                    x_val2 = helper.symbolize(x_val2, sax)
                    x_test2 = helper.symbolize(x_test2, sax)
                else:
                    x_train2 = helper.symbolizeTrans(x_train2, sax, bins = symbolsCount)
                    x_val2 = helper.symbolizeTrans(x_val2, sax, bins = symbolsCount)
                    x_test2 = helper.symbolizeTrans(x_test2, sax, bins = symbolsCount)
                print(x_train2.shape)
                #x_train1 = np.expand_dims(x_train1, axis=2)

                x_train1[:,:,i] = x_train2      
                x_val[:,:,i] = x_val2
                x_test[:,:,i] = x_test2
                
            print(x_train1.shape)
            

        else:    
            if(doSymbolify):
                scaler = StandardScaler()    
                scaler = scaler.fit(x_train1.reshape((-1,1)))
                x_train1 = scaler.transform(x_train1.reshape(-1, 1)).reshape(trainShape)
                x_val = scaler.transform(x_val.reshape(-1, 1)).reshape(valShape)
                x_test = scaler.transform(x_test.reshape(-1, 1)).reshape(testShape)

                X_test_ori = x_test.copy()
                X_val_ori = x_val.copy()
                X_train_ori = x_train1.copy()


                sax = SymbolicAggregateApproximation(n_bins=symbolsCount, strategy='uniform')
                sax.fit(x_train1)

                if(useEmbed):
                    x_train1 = helper.symbolize(x_train1, sax)
                    x_val = helper.symbolize(x_val, sax)
                    x_test = helper.symbolize(x_test, sax)
                else:
                    x_train1 = helper.symbolizeTrans(x_train1, sax, bins = symbolsCount)
                    x_val = helper.symbolizeTrans(x_val, sax, bins = symbolsCount)
                    x_test = helper.symbolizeTrans(x_test, sax, bins = symbolsCount)
            else:
                X_test_ori = x_test.copy()
                X_val_ori = x_val.copy()
                X_train_ori = x_train1.copy()


            x_train1 = np.expand_dims(x_train1, axis=2)
            x_val = np.expand_dims(x_val, axis=2)
            x_test = np.expand_dims(x_test, axis=2)   
            X_test_ori = np.expand_dims(X_test_ori, axis=2)   
            X_train_ori = np.expand_dims(X_train_ori, axis=2) 
            X_val_ori = np.expand_dims(X_val_ori, axis=2) 
            
            

        print('saves shapes:')
        print(x_test.shape)
        print(x_train1.shape)

        #save sax results to only calculate them once
        resultsSave = {
            'X_train':x_train1,
            'X_train_ori':X_train_ori,
            'X_test':x_test,
            'X_test_ori':X_test_ori,
            'X_val': x_val,
            'X_val_ori':X_val_ori,
            'y_trainy':y_trainy,
            'y_train':y_train1,
            'y_val': y_val,
            'y_test':y_test,
            'y_testy':y_testy
        }
        helper.save_obj(resultsSave, processedDataName)
    return x_train1, x_val, x_test, y_train1, y_val, y_test, X_train_ori, X_val_ori, X_test_ori, y_trainy, y_testy