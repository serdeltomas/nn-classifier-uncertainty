import numpy as np
import scipy.spatial.distance as spd

# def init_arr():
#     x = empty(10, dtype=object)
#     for i in range(10): x[i] = []
#     return x

# ------------------------------------------------------------------------------------------
def compute_dist(mav,all_ok):
    eucos = []
    for feat in all_ok:
        eucos.append(spd.euclidean(mav, feat) / 200.0 + spd.cosine(mav, feat))
    return eucos

# ------------------------------------------------------------------------------------------
def compute_dist_from_mav():
    last_layer = np.load("preprocessing/c10_train_pred.npy", allow_pickle=1)
    targets = np.load("preprocessing/c10_train_targ.npy", allow_pickle=1)
    mav = np.load("preprocessing/mav_c10_train.npy", allow_pickle=1)

    dist = []
    for category in range(10):
        correct_items = []
        for prvok, target in zip(last_layer, targets):
            if target[0] == category and target[1]:
                correct_items.append(prvok)
        dist.append(compute_dist(mav[category],correct_items))
    dist = np.asarray(dist, dtype=object)
    return dist

#------------------------------------------------------------------------------------------

if __name__ == "__main__":
    dist = compute_dist_from_mav()
    np.save("preprocessing/dist_c10_train.npy", dist)
    print("eeeey")