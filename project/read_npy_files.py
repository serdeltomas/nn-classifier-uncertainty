import numpy as np

openmax = np.load("testing/om_a%d_t%d.npy" % (1, 18), allow_pickle=True)
softmax = np.load("testing/sm.npy", allow_pickle=True)

print()