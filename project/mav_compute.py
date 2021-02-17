import numpy as np
import torch

def averagee(list):
    averagelist = []
    for i in range (10):
        sum = 0.0
        for item in list:
            sum += item[i]
        averagelist.append(sum/(1.0*len(list)))
    return averagelist

#------------------------------------------------------------------------------------------
def compute_mav():
    last_layer = np.load("preprocessing/c10_train_pred.npy", allow_pickle=1)
    targets = np.load("preprocessing/c10_train_targ.npy", allow_pickle=1)

    mav = []
    for category in range(10):
        correct_items = []
        for prvok, target in zip(last_layer, targets):
            if target[0] == category and target[1]:
                correct_items.append(prvok)
        mav.append(averagee(correct_items))
    mav = np.asarray(mav)
    return mav

#------------------------------------------------------------------------------------------

if __name__ == "__main__":
    mav = compute_mav()
    np.save("preprocessing/mav_c10_train.npy", mav)