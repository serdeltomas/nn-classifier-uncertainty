import sklearn.metrics as slm
import numpy as np
import torch
import matplotlib.pyplot as pyplot
import time

def get_TPTNFPFN_over_epsilon(sm_om, targets):
    # is_openset = np.concatenate((np.full(10000, True), np.full(10000, False)))
    tp, tn, fp, fn = [], [], [], []
    for e in range(101):
        epsilon = e/100.0
        tp_, tn_, fp_, fn_ = 0, 0, 0, 0
        for example in range(len(sm_om)):
            if np.max(sm_om[example]) < epsilon:  #nok
                if example < len(sm_om)/2.0:  #is actually OS
                    tp_ += 1
                elif example >= len(sm_om)/2.0:  #is actually CS
                    fp_ += 1
            else:  #ok
                if example < len(sm_om)/2.0:  #is actually OS
                    if np.argmax(sm_om[example]) == 10:  #guess OS
                        tp_ += 1
                    elif np.argmax(sm_om[example]) < 10:  #guess CS
                        fn_ += 1
                elif example >= len(sm_om)/2.0:  #is actually CS
                    if np.argmax(sm_om[example]) == 10:  #guess OS
                        fp_ += 1
                    elif np.argmax(sm_om[example]) < 10:  #guess CS
                        if np.argmax(sm_om[example]) == targets[example]:  #true
                            tn_ += 1
                        else:  #false
                            fn_ += 1
        assert tp_+tn_+fp_+fn_ == 20000
        tp.append(tp_)
        tn.append(tn_)
        fn.append(fn_)
        fp.append(fp_)

    return tp, tn, fp, fn
# -------------------------------------------------------------------------------------------------------------------


def evaluate(sm_om, targets, epsilon):
    sum_eval = 0
    eval_om_sm = dict(sum_eval=-1, f_measure=-1, precision=-1, recall=-1, fpr=-1, fpr95=-1, false_neg_rate=-1,
                classification_accuracy=-1, target_success_rate=-1, tp=-1, tn=-1, fp=-1, fn=-1)

    targets_sm_om = 0
    for cs_example in range(int(len(targets) / 2.0), len(targets)):
        if targets[cs_example][0] == np.argmax(sm_om[cs_example]) and sm_om[cs_example][np.argmax(sm_om[cs_example])] >= epsilon:
            targets_sm_om += 1
    eval_om_sm['classification_accuracy'] = targets_sm_om/(len(targets)/2.0)
    eval_om_sm['tp'], eval_om_sm['tn'], eval_om_sm['fp'], eval_om_sm['fn'] = get_TPTNFPFN(sm_om, targets, epsilon)
    # fn/(fn+tp)
    if not eval_om_sm['fn'] and not eval_om_sm['tp']:
        eval_om_sm['false_neg_rate'] = 0
    else:
        eval_om_sm['false_neg_rate'] = eval_om_sm['fn'] / (eval_om_sm['fn'] + eval_om_sm['tp'])
    # fp/(fp+tn)
    if not eval_om_sm['fp'] and not eval_om_sm['tn']:
        eval_om_sm['fpr'] = 0
    else:
        eval_om_sm['fpr'] = eval_om_sm['fp'] / (eval_om_sm['fp'] + eval_om_sm['tn'])
    # tp/(tp+fp)
    if not eval_om_sm['tp'] and not eval_om_sm['fp']:
        eval_om_sm['precision'] = 0
    else:
        eval_om_sm['precision'] = eval_om_sm['tp'] / (eval_om_sm['tp'] + eval_om_sm['fp'])
    # tp/(tp+fn)
    if not eval_om_sm['tp'] and not eval_om_sm['fn']:
        eval_om_sm['recall'] = 0
    else:
        eval_om_sm['recall'] = eval_om_sm['tp'] / (eval_om_sm['tp'] + eval_om_sm['fn'])
    # recall=1-fnr
    assert 1 == eval_om_sm['recall'] + eval_om_sm['false_neg_rate']
    # 2*precision*recall/(precision+recall)
    if not eval_om_sm['precision'] and not eval_om_sm['recall']:
        eval_om_sm['f_measure'] = 0
    else:
        eval_om_sm['f_measure'] = 2 * eval_om_sm['precision'] * eval_om_sm['recall'] / (eval_om_sm['precision'] + eval_om_sm['recall'])

    eval_om_sm['sum_eval'] = (eval_om_sm['f_measure'] + eval_om_sm['classification_accuracy'] + (1 - eval_om_sm['false_neg_rate'])) * (100.0/3.0)

    print("", end="")
    return eval_om_sm  #sum_eval is 0-100 with 100 being the best
# -------------------------------------------------------------------------------------------------------------------


def compute_auroc_old(om_sm, targets):
    tp, tn, fp, fn = get_TPTNFPFN_over_epsilon(om_sm,targets)
    fpr, tpr = [], []
    for i in range(101):
        if not fp[i] and not tn[i]:
            fpr.append(0)
        else:
            fpr.append(fp[i]/(fp[i] + tn[i]))
        if not tp[i] and not fn[i]:
            tpr.append(0)
        else:
            tpr.append(tp[i] / (tp[i] + fn[i]))

    auroc = slm.auc(fpr, tpr)
    return auroc
# -------------------------------------------------------------------------------------------------------------------


def compute_auroc(is_positive, is_rejected):
    fpr, tpr = [], []

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


def is_rejected_one(position, value, epsilon):
    return position == 10 or value < epsilon
# -------------------------------------------------------------------------------------------------------------------


def is_rejected_over_epsilon(position, value):
    is_rej = []
    for epsilon in range(101):
        is_rej.append((position == 10 or value < epsilon/100.0))
    return np.asarray(is_rej)
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


def is_positive_all(targets):
    is_pos_all = []
    for targ in targets:
        if targ == 10:
            is_pos_all.append(True)
        else:
            is_pos_all.append(False)
    return np.asarray(is_pos_all)
# -------------------------------------------------------------------------------------------------------------------


def save_rejected():
    for tail_size in range(2, 22):
        for alpha in range(1, 7):
            start_time = time.time()
            om = np.load("testing/om_a%d_t%d.npy" % (alpha, tail_size), allow_pickle=True)
            om_topk = torch.topk(torch.from_numpy(om), k=1, dim=1)
            is_rej_om = is_rejected_all(om_topk)
            np.save("testing/rejected/om_rej_a%d_t%d.npy" % (alpha, tail_size), is_rej_om)
            print("--- %s seconds ---" % (time.time() - start_time))
    return True
# -------------------------------------------------------------------------------------------------------------------


def main_grid_search():
    # x, y = np.hsplit(np.load("preprocessing/c10_test_targ.npy", allow_pickle=True), 2)
    # targets = np.concatenate((np.full(10000, 10), x.flatten()))
    is_positive = np.concatenate((np.full(10000, True), np.full(10000, False)))
    auroc_om_all = np.zeros((24, 6))

    counter = 0
    for tail_size in range(2, 26):
        for alpha in range(1, 7):
            start_time = time.time()
            is_rejected = np.load("testing/rejected/om_rej_a%d_t%d.npy" % (alpha, tail_size), allow_pickle=True)
            auroc_om_all[tail_size - 2][alpha - 1] = compute_auroc(is_positive, is_rejected)
            # auroc_sm_all[tail_size - 2][alpha - 1] = compute_auroc(is_positive, is_rejected)
            counter += 1
            print("%s/50 " % (str(counter).zfill(3)), end="")
            print("--- %s seconds ---" % (time.time() - start_time))
    print(auroc_om_all)
    return 0
# -------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    main_grid_search()
    # save_rejected()
