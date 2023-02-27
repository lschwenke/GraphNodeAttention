import numpy as np
from modules import helper
import math
from scipy.interpolate import interp1d

def validataHeat(value, heat, doFidelity):
    if doFidelity:
        return value <= heat
    else:
        return value > heat

#interpolation with customized combinations
def abstractDataS(data, earlyPredictorZ, order, step1, step2, step3, doMax, thresholdSet, useEmbed = False, takeAvg = True, heatLayer = 0, interpolate = True, doFidelity=False, limit = 500):
    

    #aggregate attention vector
    attentionQ0 = []
    attentionQ1 = []
    attentionQ2 = []

    for bor in range(int(math.ceil(len(data)/limit))):
        attOut = earlyPredictorZ.predict([data[bor*limit:(bor+1)*limit]])
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
        axis1 = 2
        axis2 = 0  

    print('*********************')
    print(attentionFQ[2].shape)
    attentionFQ[1] = helper.doCombiStep(step1, attentionFQ[2], axis1)
    print(attentionFQ[1].shape)
    attentionFQ[1] = helper.doCombiStep(step2, attentionFQ[1], axis2) 
    print(attentionFQ[1].shape)

    print(attentionFQ[1].shape)

    # create new data based on some threshold(s)
    newX = []
    reduction = []
    skipCounter = 0 
    for index in range(len(attentionFQ[1])):
                   
            if(useEmbed):
                X_sax = np.array(data).squeeze()[index].split(" ")
                vocab = helper.getMapValues(len(set(X_sax)))
                X_ori1 = [helper.trans(valx, vocab) for valx in np.array(data).squeeze()[index].split(" ")]
            else:
                X_sax = np.array(data[0])[index]
                X_ori1 = X_sax 
  
            heat = helper.doCombiStep(step3, attentionFQ[1][index], 0) 
    
            if doMax:
                maxHeat = np.max(heat)
                borderHeat = maxHeat/thresholdSet[0]   #/2
                borderHeat2 = maxHeat/thresholdSet[1]#/3
            else:
                maxHeat = np.average(heat)
                borderHeat = maxHeat/thresholdSet[0]
                borderHeat2 = maxHeat/thresholdSet[1]#/1.2

            strongerInterpolation = True
            mediumSkips = 0

            if(doFidelity):  
                bufferHeat = borderHeat
                borderHeat = borderHeat2
                borderHeat2 = bufferHeat
            
            # interpolate missing data or fill the missing parts with -2 to enable masking
            if interpolate:
                newXT = []

                for i in range(X_ori1.shape[-1]):
                    newX2 = []
                    X_ori = X_ori1[:,i]
                    fitleredSet = []
                    indexSet = []
                    avgSet = []
                    for h in range(len(heat)):
                        if validataHeat(heat[h], borderHeat, doFidelity):
                            if len(avgSet) > mediumSkips:
                                fitleredSet.append(np.median(avgSet))
                                indexSet.append(h - math.ceil(len(avgSet)/2))
                                avgSet = []
                            fitleredSet.append(X_ori[h])
                            indexSet.append(h)
                        elif validataHeat(heat[h], borderHeat2, doFidelity):
                            #fitleredSet.append([-1e9])
                            avgSet.append(X_ori[h])
                            #avgSet = []
                        elif len(avgSet) > mediumSkips and strongerInterpolation:
                            #fitleredSet.append([-1e9])
                            fitleredSet.append(np.median(avgSet))
                            indexSet.append(h - math.ceil(len(avgSet)/2))

                            avgSet = []
                        #else:
                            #fitleredSet.append([-1e9])

                    if len(avgSet) > mediumSkips:
                        fitleredSet.append(np.median(avgSet))
                        indexSet.append(len(heat) - math.ceil(len(avgSet)/2))

                    reduction.append(1 - len(fitleredSet)/len(heat))

                    if(len(fitleredSet) == 0):
                        skipCounter += 1
                        fitleredSet.append(0)
                        indexSet.append(0)
                        fitleredSet.append(0)
                        indexSet.append(len(heat))
                    elif(len(fitleredSet) < 2):
                        skipCounter += 1
                        fitleredSet.append(0)
                        indexSet.append(len(heat))
                    newXTemp = interp1d(indexSet, fitleredSet, bounds_error = False, fill_value = -2)
                    newX2.append([x for x in newXTemp(range(len(heat)))])
                    newX2 = np.array(newX2, dtype=np.float32)
                    newXT.append(newX2[0])

                newXT = np.array(newXT, dtype=np.float32)
                newX.append(np.transpose(newXT, (1,0)))
            
            else:                 
                newXT = []
                for i in range(X_ori1.shape[-1]):
                    fitleredSet = []
                    indexSet = []
                    avgSet = []
                    newX2 = []
                    X_ori = X_ori1[:,i]
                    for h in range(len(heat)):
                        if validataHeat(heat[h], borderHeat, doFidelity):
                            if len(avgSet) != 0:
                                fitleredSet[h - math.ceil(len(avgSet)/2)] = np.median(avgSet)
                                avgSet = []
                            fitleredSet.append(X_ori[h])
                        elif validataHeat(heat[h], borderHeat2, doFidelity):
                            fitleredSet.append(-2)
                            avgSet.append(X_ori[h])
                        elif len(avgSet) != 0:
                            fitleredSet.append(-2)
                            fitleredSet[h - math.ceil(len(avgSet)/2)] = np.median(avgSet)

                            avgSet = []
                        else:
                            fitleredSet.append(-2)
                    if len(avgSet) != 0:
                        fitleredSet[len(heat) - math.ceil(len(avgSet)/2)] = np.median(avgSet)
                    reduction.append(1 - len([x for x in fitleredSet if x != -2])/len(heat))
                    newX2.append([x for x in fitleredSet])
                    newX2 = np.array(newX2, dtype=np.float32)
                    newXT.append(newX2[0])
                newXT = np.array(newXT, dtype=np.float32)
                newX.append(np.transpose(newXT, (1,0)))

    newX = np.array(newX, dtype=np.float32)
    print(np.array(newX).shape)
    print(X_ori1.shape)

    return newX, reduction, skipCounter
