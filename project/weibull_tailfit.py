import numpy as np
# from openmax_utils import *
import libmr


# ---------------------------------------------------------------------------------
def weibull_tailfit(tailsize):
    dist = np.load("preprocessing/dist_c10_train.npy", allow_pickle=1)

    weibull_model = []
    for category in range(10):
            mr = libmr.MR()
            tailtofit = sorted(dist[category])[-tailsize:]
            mr.fit_high(tailtofit, len(tailtofit))
            weibull_model.append(mr)

    weibull_model = np.asarray(weibull_model)
    return weibull_model

# ---------------------------------------------------------------------------------
if __name__ == "__main__":
    weibull = weibull_tailfit(20)
    # np.save("preprocessing/weibull_models.npy", weibull)

