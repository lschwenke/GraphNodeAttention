import numpy as np
import dill as pickle
import math

def doCombiStep(step, field, axis) -> np.ndarray:
    if(step == 'max'):
        return np.max(field, axis=axis)
    elif (step == 'sum'):
        return np.sum(field, axis=axis)

#flatten an 3D np array
def flatten(X, pos = -1):
    flattened_X = np.empty((X.shape[0], X.shape[2]))  # sample x features array.
    if pos == -1:
        pos = X.shape[1]-1
    for i in range(X.shape[0]):
        flattened_X[i] = X[i, pos, :]
    return(flattened_X)

# Scale 3D array. X = 3D array, scalar = scale object from sklearn. Output = scaled 3D array.
def scale(X, scaler):
    for i in range(X.shape[0]):
        X[i, :, :] = scaler.transform(X[i, :, :])
    return X

# Symbolize a 3D array. X = 3D array, scalar = SAX symbolizer object. Output = symbolic 3D string array.
def symbolize(X, scaler):
    X_s = scaler.transform(X)
    #X_s = X.astype('U13') 
    #for i in range(X.shape[0]):
        #X_s[i, :, :][:,0] = scaler.transform(np.array([X[i, :, :][:,0]]))
        #X_t.append(' '.join(X_s[i, :, :][:,0]))
    return X_s

# translate the a string [a,e] between 
def trans(val, vocab) -> float:
    for i in range(len(vocab)):
        if val == vocab[i]:
            halfSize = (len(vocab)-1)/2
            return (i - halfSize) / halfSize
    return -2

def getMapValues(size):
    vMap = []
    for i in range(size):
        halfSize = (size-1)/2
        vMap.append((i - halfSize) / halfSize)
    return vMap

def symbolizeTrans(X, scaler, bins = 5):
    vocab = scaler._check_params(bins)
    X_s = scaler.transform(X)
    #X_s = X.astype(str) 
    for i in range(X.shape[0]):
        X = X.astype(float)
        
        #z1 = scaler.transform(np.array([X[i, :, :][:,0]]))
        
        for j in range(X.shape[1]):
            X[i][j] = trans(X_s[i][j], vocab)
    return X

def symbolizeTrans2(X, scaler, bins = 5):
    vocab = scaler._check_params(bins)
    for i in range(X.shape[0]):
        #X = X.astype('U13')
        X_s = X.astype(str) 
        z1 = scaler.transform(np.array([X[i, :, :][:,0]]))
        X_s[i, :, :][:,0] = z1
        for j in range(X.shape[1]):
            X[i][j][0] = trans(X_s[i][j][0], vocab)
    return X

def split_dataframe(df, chunk_size = 10000): 
    chunks = list()
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        chunks.append(df[i*chunk_size:(i+1)*chunk_size])
    return chunks

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        np.save(f, obj)

def save_obj2(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(f, obj)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return np.load(f, allow_pickle=True)
		
def load_obj2(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
def truncate(n):
    return int(n * 1000) / 1000

def ceFull(data):
    complexDists = []
    for d in data:
        complexDists.append(ce(d))
        
    return complexDists, np.mean(complexDists) 

def ce(data):
    summer = 0
    for i in range(len(data)-1):
        summer += math.pow(data[i] - data[i+1], 2)
    return math.sqrt(summer)


def modelFidelity(modelPrediction, interpretationPrediction):
    summer = 0
    for i in range(len(modelPrediction)):
        if modelPrediction[i] == interpretationPrediction[i]:
            summer += 1
    return summer / len(modelPrediction)

def collectLAAMs(earlyPredictor, x_test, order, step1, step2):
    limit = 500
    attentionQ0 = []
    attentionQ1 = []
    attentionQ2 = []

    for bor in range(int(math.ceil(len(x_test)/limit))):
        attOut = earlyPredictor.predict([x_test[bor*limit:(bor+1)*limit]])
        attentionQ0.extend(attOut[0]) 
        attentionQ1.extend(attOut[1])

        if len(attentionQ2) == 0:
            attentionQ2 = attOut[2]
        else:
            print(np.array(attentionQ2).shape)
            for k in range(len(attentionQ2)):
                
                attentionQ2[k] = np.append(attentionQ2[k], attOut[2][k], 0)
            print(np.array(attentionQ2).shape)
    
    attentionFQ = [np.array(attentionQ0), np.array(attentionQ1), np.array(attentionQ2)]
    
    if(order == 'lh'):
        axis1 = 0
        axis2 = 1
    elif(order == 'hl'):
        axis1 = 2
        axis2 = 0
    else:
        axis1 = 0
        axis2 = 1   

    attentionFQ[1] = doCombiStep(step1, attentionFQ[2], axis1)
    attentionFQ[1] = doCombiStep(step2, attentionFQ[1], axis2) 

    return attentionFQ[1]


def laamConsistency(laams, fromIndex, consistancyLabels):
    results = dict()
    innerFolddistance = dict()
    outerdistance = dict()
    innerClassDistance = dict()
    for combi in laams.keys():
        innerFolddistance[combi] = []
        outerdistance[combi] = []
        innerClassDistance[combi] = []
        for j in range(len(laams[combi][fromIndex])):
            outerdistance[combi].append([])
            for fold in range(len(laams[combi])):
                if fold != fromIndex:
                    outerdistance[combi][j].append(matrixEucDistance(laams[combi][fromIndex][j],laams[combi][fold][j]))

        for fold in range(len(laams[combi])): 
            innerFolddistance[combi].append([])
            for j in range(len(laams[combi][fold])):
                if j != fromIndex and consistancyLabels[j] != consistancyLabels[fromIndex]:
                    innerFolddistance[combi][fold].append(matrixEucDistance(laams[combi][fold][fromIndex],laams[combi][fold][j]))
        
        for fold in range(len(laams[combi])): 
            innerClassDistance[combi].append([])
            for j in range(len(laams[combi][fold])):
                if j != fromIndex and consistancyLabels[j] == consistancyLabels[fromIndex]:
                    innerClassDistance[combi][fold].append(matrixEucDistance(laams[combi][fold][fromIndex],laams[combi][fold][j]))
    
    #from one sample to another of a different class
    results["innerFold"] = innerFolddistance
    #from one sample to another of the same class
    results["innerClass"] = innerClassDistance
    #same sample between folds
    results["outer"] = outerdistance
    return results

def matrixEucDistance(matrix1, matrix2):
    summer = 0
    for i in range(len(matrix1)):
        for j in range(len(matrix1[i])):
            summer += math.pow(matrix1[i][j] - matrix2[i][j], 2)
    return math.sqrt(summer)

def confidenceGCR(bestScores, correctness):
    top80Len = int(len(bestScores) * 0.80)
    top50Len = int(len(bestScores) * 0.50)
    top20Len = int(len(bestScores) * 0.20)
    top10Len = int(len(bestScores) * 0.10)
    
    top80Ind = sorted(range(len(bestScores)), key=lambda x: bestScores[x])[-top80Len:]
    top50Ind = sorted(range(len(bestScores)), key=lambda x: bestScores[x])[-top50Len:]
    top20Ind = sorted(range(len(bestScores)), key=lambda x: bestScores[x])[-top20Len:]
    top10Ind = sorted(range(len(bestScores)), key=lambda x: bestScores[x])[-top10Len:]

    nCorrectness = np.array(correctness)
    top80Acc = (sum(nCorrectness[top80Ind])/len(nCorrectness[top80Ind]))
    top50Acc = (sum(nCorrectness[top50Ind])/len(nCorrectness[top50Ind]))
    top20Acc = (sum(nCorrectness[top20Ind])/len(nCorrectness[top20Ind]))
    top10Acc = (sum(nCorrectness[top10Ind])/len(nCorrectness[top10Ind]))
    
    return top80Acc, top50Acc, top20Acc, top10Acc

def fidelityConfidenceGCR(bestScores, correctness, saxResults):
    top80Len = int(len(bestScores) * 0.80)
    top50Len = int(len(bestScores) * 0.50)
    top20Len = int(len(bestScores) * 0.20)
    top10Len = int(len(bestScores) * 0.10)
    
    top80Ind = sorted(range(len(bestScores)), key=lambda x: bestScores[x])[-top80Len:]
    top50Ind = sorted(range(len(bestScores)), key=lambda x: bestScores[x])[-top50Len:]
    top20Ind = sorted(range(len(bestScores)), key=lambda x: bestScores[x])[-top20Len:]
    top10Ind = sorted(range(len(bestScores)), key=lambda x: bestScores[x])[-top10Len:]

    nCorrectness = np.array(correctness)
    nSaxResults = np.array(saxResults)
    top80Fidelity = modelFidelity(nCorrectness[top80Ind], nSaxResults[top80Ind])
    top50Fidelity = modelFidelity(nCorrectness[top50Ind], nSaxResults[top50Ind])
    top20Fidelity = modelFidelity(nCorrectness[top20Ind], nSaxResults[top20Ind])
    top10Fidelity = modelFidelity(nCorrectness[top10Ind], nSaxResults[top10Ind])


    return top80Fidelity, top50Fidelity, top20Fidelity, top10Fidelity