import sklearn.metrics as slm
import numpy as np
import matplotlib.pyplot as pyplot


def get_TPTNFPFN(sm_om, targets, epsilon):
    tp, tn, fp, fn = 0, 0, 0, 0

    for example in range(len(sm_om)):
        if np.max(sm_om[example]) < epsilon:  #nok
            if example < len(sm_om)/2.0:  #is actually OS
                tn += 1
            elif example >= len(sm_om)/2.0:  #is actually CS
                fn += 1
        else:  #ok
            if example < len(sm_om)/2.0:  #is actually OS
                if np.argmax(sm_om[example]) == 10:  #guess OS
                    tn += 1
                elif np.argmax(sm_om[example]) < 10:  #guess CS
                    fp += 1
            elif example >= len(sm_om)/2.0:  #is actually CS
                if np.argmax(sm_om[example]) == 10:  #guess OS
                    fn += 1
                elif np.argmax(sm_om[example]) < 10:  #guess CS
                    if np.argmax(sm_om[example]) == targets[example][0]:  #true
                        tp += 1
                    else:  #false
                        fp += 1

    assert tp+tn+fp+fn == len(sm_om)
    # print(tp+tn+fp+fn, end="")
    # print("/",end="")
    # print(len(sm_om))
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


def main_grid_search():
    targets = np.concatenate((np.load("preprocessing/c100_test_targ.npy", allow_pickle=True),
                              np.load("preprocessing/c10_test_targ.npy", allow_pickle=True)), axis=0)
    openmax_all, softmax_all = [], []

    openmax_best = dict(sum_eval=-1, eval=-1, alpha=-1, tail=-1, epsilon=-1)
    softmax_best = dict(sum_eval=-1, eval=-1, alpha=-1, tail=-1, epsilon=-1)
    both_best = dict(sum_eval=-1, eval_o=-1, eval_s=-1, alpha=-1, tail=-1, epsilon=-1)
    openmax_best_auroc = dict(auroc=-1, alpha=-1, tail=-1)
    softmax_best_auroc = dict(auroc=-1, alpha=-1, tail=-1)
    both_best_auroc = dict(auroc=-1, alpha=-1, tail=-1)

    for i in range(int(len(targets) / 2)):
        targets[i][0] = 10

    for tail_size in range(1, 31):
        for alpha in range(1, 9):
            auroc_om, aurooc_sm, auroc_both = 0, 0, 0
            fpr_om, fpr_sm, tpr_om, tpr_sm = [], [], [], []
            openmax = np.load("testing/om_a%d_t%d.npy" %(alpha, tail_size), allow_pickle=True)
            softmax = np.load("testing/sm_a%d_t%d.npy" %(alpha, tail_size), allow_pickle=True)
            for epsilon in range(0, 101):
                sm_eval = evaluate(np.asarray(softmax), targets, epsilon / 100.0)
                om_eval = evaluate(np.asarray(openmax), targets, epsilon / 100.0)

                openmax_ = dict(sum_eval=-1, eval=-1, alpha=-1, tail=-1, epsilon=-1)
                softmax_ = dict(sum_eval=-1, eval=-1, alpha=-1, tail=-1, epsilon=-1)
                softmax_['sum_eval'], softmax_['eval'] = sm_eval['sum_eval'], sm_eval
                softmax_['alpha'], softmax_['tail'], softmax_['epsilon'] = alpha, tail_size, epsilon
                openmax_['sum_eval'], openmax_['eval'] = om_eval['sum_eval'], om_eval
                openmax_['alpha'], openmax_['tail'], openmax_['epsilon'] = alpha, tail_size, epsilon
                openmax_all.append(openmax_)
                softmax_all.append(softmax_)
                fpr_om.append(om_eval['fpr'])
                fpr_sm.append(sm_eval['fpr'])
                tpr_om.append(om_eval['recall'])
                tpr_sm.append(sm_eval['recall'])
                if sm_eval['sum_eval'] > softmax_best['sum_eval']:
                    softmax_best['sum_eval'], softmax_best['eval'] = sm_eval['sum_eval'], sm_eval
                    softmax_best['alpha'], softmax_best['tail'], softmax_best['epsilon'] = alpha, tail_size, epsilon
                if om_eval['sum_eval'] > openmax_best['sum_eval']:
                    openmax_best['sum_eval'], openmax_best['eval'] = om_eval['sum_eval'], om_eval
                    openmax_best['alpha'], openmax_best['tail'], openmax_best['epsilon'] = alpha, tail_size, epsilon
                if (om_eval['sum_eval'] + sm_eval['sum_eval'])/2.0 > both_best['sum_eval']:
                    both_best['sum_eval'] = (om_eval['sum_eval'] + sm_eval['sum_eval'])/2.0
                    both_best['eval_o'], both_best['eval_s'] = om_eval, sm_eval
                    both_best['alpha'], both_best['tail'], both_best['epsilon'] = alpha, tail_size, epsilon
            auroc_om = slm.auc(fpr_om, tpr_om)
            auroc_sm = slm.auc(fpr_sm, tpr_sm)
            auroc_both = (auroc_sm + auroc_om) / 2.0
            # pyplot.plot(fpr_om, tpr_om)
            # pyplot.plot(fpr_sm, tpr_sm)
            if auroc_sm > softmax_best_auroc['auroc']:
                softmax_best_auroc['auroc'] = auroc_sm
                softmax_best_auroc['alpha'], softmax_best_auroc['tail'] = alpha, tail_size
            if auroc_om > openmax_best_auroc['auroc']:
                openmax_best_auroc['auroc'] = auroc_om
                openmax_best_auroc['alpha'], openmax_best_auroc['tail'] = alpha, tail_size
            if auroc_both > both_best_auroc['auroc']:
                both_best_auroc['auroc'] = auroc_both
                both_best_auroc['alpha'], both_best_auroc['tail'] = alpha, tail_size

    openmax_all = np.asarray(openmax_all)
    softmax_all = np.asarray(softmax_all)
    np.save("results/openmax_test_results_a1-8_t1-30_e0-100.npy", openmax_all)
    np.save("results/softmax_test_results_a1-8_t1-30_e0-100.npy", softmax_all)
    print(openmax_best)
    print(softmax_best)
    print(both_best)
    print(openmax_best_auroc)
    print(softmax_best_auroc)
    print(both_best_auroc)
    return 0
# -------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    main_grid_search()