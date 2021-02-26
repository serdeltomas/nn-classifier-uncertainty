import sklearn.metrics as slm
import numpy as np
import torch
import matplotlib.pyplot as pyplot
import time


def compute_auroc(is_positive, is_rejected):
    # computes auroc for one combination of tail and alpha
    fpr, tpr = [], []  # tpr=recall

    for epsilon in range(101):
        tp, tn, fp, fn = 0, 0, 0, 0
        for is_pos, is_rej in zip(is_positive, is_rejected[epsilon]):
            if is_rej and is_pos:
                tp += 1
            elif not is_rej and not is_pos:
                tn += 1
            elif is_rej and not is_pos:
                fp += 1
            elif not is_rej and is_pos:
                fn += 1

        assert tp+tn+fp+fn == 20000

        if not fp and not tn:
            fpr.append(0)
        else:
            fpr.append(fp / (fp + tn))
        if not tp and not fn:
            tpr.append(0)
        else:
            tpr.append(tp / (tp + fn))

    auroc = slm.auc(fpr, tpr)
    return auroc
# -------------------------------------------------------------------------------------------------------------------


def compute_aupr(is_positive, is_rejected):
    # computes aupr for one combination of tail and alpha
    precision, recall = [], []  # tpr=recall

    for epsilon in range(101):
        tp, tn, fp, fn = 0, 0, 0, 0
        for is_pos, is_rej in zip(is_positive, is_rejected[epsilon]):
            if is_rej and is_pos:
                tp += 1
            elif not is_rej and not is_pos:
                tn += 1
            elif is_rej and not is_pos:
                fp += 1
            elif not is_rej and is_pos:
                fn += 1

        assert tp+tn+fp+fn == 20000

        if not tp and not fp:
            precision.append(0)
        else:
            precision.append(tp / (tp + fp))
        if not tp and not fn:
            recall.append(0)
        else:
            recall.append(tp / (tp + fn))

    aupr = slm.auc(recall, precision)
    return aupr
# -------------------------------------------------------------------------------------------------------------------



def compute_precision_recall(is_positive, is_rejected):
    # computes precision and recall for one combination of tail and alpha over epsilon
    precision, recall = [], []  # tpr=recall

    for epsilon in range(101):
        tp, tn, fp, fn = 0, 0, 0, 0
        for is_pos, is_rej in zip(is_positive, is_rejected[epsilon]):
            if is_rej and is_pos:
                tp += 1
            elif not is_rej and not is_pos:
                tn += 1
            elif is_rej and not is_pos:
                fp += 1
            elif not is_rej and is_pos:
                fn += 1

        assert tp+tn+fp+fn == 20000

        if not tp and not fp:
            precision.append(0)
        else:
            precision.append(tp / (tp + fp))
        if not tp and not fn:
            recall.append(0)
        else:
            recall.append(tp / (tp + fn))

    return precision, recall
# -------------------------------------------------------------------------------------------------------------------


def is_rejected_all(topk):
    is_rej_all = []
    for topk_val, topk_pos in zip(topk[0], topk[1]):
        is_rej = []
        for epsilon in range(101):
            is_rej.append((topk_pos == 10 or topk_val < (epsilon / 100.0)))
        is_rej_all.append(np.asarray(is_rej))
    is_rej_all = np.asarray(is_rej_all)
    return is_rej_all.T
# -------------------------------------------------------------------------------------------------------------------


def classification_accuracy(targets, predictions):
    # sm = np.load("testing/sm.npy", allow_pickle=True)
    # om = np.load("testing/om_a%d_t%d.npy" % (alpha,tail), allow_pickle=True)

    ca = 0
    for i in range(10000, 20000):
        if targets[i] == np.argmax(predictions[i]):
            ca += 1

    ca = ca/10000.0
    return ca
# -------------------------------------------------------------------------------------------------------------------


def classification_accuracy_epsilon(targets, predictions, epsilon):
    # sm = np.load("testing/sm.npy", allow_pickle=True)
    # om = np.load("testing/om_a%d_t%d.npy" % (alpha,tail), allow_pickle=True)

    ca = 0
    for i in range(10000, 20000):
        argm = np.argmax(predictions[i])
        max = predictions[i][argm]
        if max >= epsilon and targets[i] == argm:  # zamietni ak epsilon je ostro väčšie
            ca += 1

    ca = ca/10000.0
    return ca
# -------------------------------------------------------------------------------------------------------------------


def save_rejected():
    for tail_size in range(2, 22):
        for alpha in range(1, 7):
            # start_time = time.time()
            om = np.load("testing/om_a%d_t%d.npy" % (alpha, tail_size), allow_pickle=True)
            om_topk = torch.topk(torch.from_numpy(om), k=1, dim=1)
            is_rej_om = is_rejected_all(om_topk)
            # np.save("testing/rejected/om_rej_a%d_t%d.npy" % (alpha, tail_size), is_rej_om)
            # print("--- %s seconds ---" % (time.time() - start_time))
    return True
# -------------------------------------------------------------------------------------------------------------------


def grid_auroc():
    # x, y = np.hsplit(np.load("preprocessing/c10_test_targ.npy", allow_pickle=True), 2)
    # targets = np.concatenate((np.full(10000, 10), x.flatten()))
    is_positive = np.concatenate((np.full(10000, True), np.full(10000, False)))
    auroc_om_all = np.zeros((24, 6))
    is_rejected_sm = np.load("testing/rejected/sm_rej.npy", allow_pickle=True)
    auroc_sm = compute_auroc(is_positive, is_rejected_sm)
    counter = 0
    for tail_size in range(2, 26):
        for alpha in range(1, 7):
            # start_time = time.time()
            is_rejected = np.load("testing/rejected/om_rej_a%d_t%d.npy" % (alpha, tail_size), allow_pickle=True)
            auroc_om_all[tail_size - 2][alpha - 1] = compute_auroc(is_positive, is_rejected)
            counter += 1
            # print("%s " % (str(counter).zfill(3)), end="")
            # print("--- %s seconds ---" % (time.time() - start_time))
    print(auroc_om_all)
    print(auroc_sm)
    return auroc_om_all, auroc_sm
# -------------------------------------------------------------------------------------------------------------------


def grid_aupr():
    is_positive = np.concatenate((np.full(10000, True), np.full(10000, False)))
    aupr_om_all = np.zeros((24, 6))
    is_rejected_sm = np.load("testing/rejected/sm_rej.npy", allow_pickle=True)
    aupr_sm = compute_aupr(is_positive, is_rejected_sm)
    counter = 0
    for tail_size in range(2, 26):
        for alpha in range(1, 7):
            # start_time = time.time()
            is_rejected = np.load("testing/rejected/om_rej_a%d_t%d.npy" % (alpha, tail_size), allow_pickle=True)
            aupr_om_all[tail_size - 2][alpha - 1] = compute_aupr(is_positive, is_rejected)
            counter += 1
            # print("%s " % (str(counter).zfill(3)), end="")
            # print("--- %s seconds ---" % (time.time() - start_time))
    print(aupr_om_all)
    print(aupr_sm)
    return aupr_om_all, aupr_sm
# -------------------------------------------------------------------------------------------------------------------


def grid_precision_recall():
    is_positive = np.concatenate((np.full(10000, True), np.full(10000, False)))
    sm_rej = np.load("testing/rejected/sm_rej.npy", allow_pickle=True)
    pr_om_all, re_om_all = [], []
    # pr_sm_all, re_sm_all = 0, 0
    alpha = [3, 2, 1]
    tail = [3, 7, 2]

    pr_sm_all, re_sm_all = compute_precision_recall(is_positive, sm_rej)
    for al_ta in range(3):
        om_rej = np.load("testing/rejected/om_rej_a%d_t%d.npy" % (alpha[al_ta], tail[al_ta]), allow_pickle=True)
        pr_om, re_om = compute_precision_recall(is_positive, om_rej)
        pr_om_all.append(pr_om)
        re_om_all.append(re_om)

    # numpy_array = np.array(pr_om_all)
    # transpose = numpy_array.T
    # pr_om_all = transpose.tolist()
    # numpy_array = np.array(re_om_all)
    # transpose = numpy_array.T
    # re_om_all = transpose.tolist()

    print(pr_om_all)
    print(re_om_all)
    print(pr_sm_all)
    print(re_sm_all)
    return 0
# -------------------------------------------------------------------------------------------------------------------


def grid_ca():
    x, y = np.hsplit(np.load("preprocessing/c10_test_targ.npy", allow_pickle=True), 2)
    targets = np.concatenate((np.full(10000, 10), x.flatten()))
    sm = np.load("testing/sm.npy", allow_pickle=True)
    ca_om = np.zeros((29, 8))
    ca_sm = classification_accuracy(targets, sm)
    # counter = 0
    for tail_size in range(2, 31):
        for alpha in range(1, 9):
            om = np.load("testing/om_a%d_t%d.npy" % (alpha,tail_size), allow_pickle=True)
            ca_om[tail_size - 2][alpha - 1] = classification_accuracy(targets, om)
            # counter += 1
            # print("%s " % (str(counter).zfill(3)), end="")
            # print("--- %s seconds ---" % (time.time() - start_time))
    print(ca_om)
    print(ca_sm)
    return ca_om, ca_sm
# -------------------------------------------------------------------------------------------------------------------


def grid_ca_eps():
    x, y = np.hsplit(np.load("preprocessing/c10_test_targ.npy", allow_pickle=True), 2)
    targets = np.concatenate((np.full(10000, 10), x.flatten()))
    sm = np.load("testing/sm.npy", allow_pickle=True)
    ca_om_all, ca_sm_all = [], []
    alpha = [3, 2, 1]
    tail = [3, 7, 2]
    for eps in range(0, 101):
        ca_om = np.zeros(3)
        ca_sm = classification_accuracy_epsilon(targets, sm, eps/100.0)
        # counter = 0
        for al_ta in range(3):
            om = np.load("testing/om_a%d_t%d.npy" % (alpha[al_ta], tail[al_ta]), allow_pickle=True)
            ca_om[al_ta] = classification_accuracy_epsilon(targets, om, eps/100.0)
            # counter += 1
            # print("%s " % (str(counter).zfill(3)), end="")
            # print("--- %s seconds ---" % (time.time() - start_time))
        ca_sm_all.append(ca_sm)
        ca_om_all.append(ca_om)
    # ca_om_all = np.asarray(ca_om_all)
    # ca_sm_all = np.asarray(ca_sm_all)

    numpy_array = np.array(ca_om_all)
    transpose = numpy_array.T
    ca_om_all = transpose.tolist()

    print(ca_om_all)
    print(ca_sm_all)
    return ca_om_all, ca_sm_all
# -------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    # grid_auroc()
    # grid_aupr()
    # save_rejected()
    # grid_ca_eps()
    # grid_ca_eps()
    grid_precision_recall()
