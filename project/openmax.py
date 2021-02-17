import numpy as np

from weibull_tailfit import weibull_tailfit
from dist_compute import compute_dist

TAIL_SIZE = 2
APLHA_RANK = 1
# IMG_INDEX = 396  # bear: 396, 413, 493
# IMG_INDEX = 226  # apple: 4, 113, 226, 377
# IMG_INDEX = 10003  # airplane: 3, 10, 21, 27, 44 (+10000)
IMG_INDEX = 10044  # doggo: 12, 16, 24, 33 (+10000)


def main(index_of_img_in_file, alpha=APLHA_RANK, tail_size=TAIL_SIZE):
    # last_layer = np.load("preprocessing/train_output_predictions.npy", allow_pickle=True) #for use with main_nok
    last_layer = np.concatenate((np.load("preprocessing/c100_test_pred.npy", allow_pickle=True),
                                 np.load("preprocessing/c10_test_pred.npy", allow_pickle=True)), axis=0)
    mav = np.load("preprocessing/mav_c10_train.npy", allow_pickle=True)
    # distance_to_mav = np.load("preprocessing/dist_from_mav.npy", allow_pickle=1)
    img = last_layer[index_of_img_in_file]
    weibull_model = weibull_tailfit(tail_size)

    openmax_probab = compute_openmax(mav, weibull_model, img, alpha)
    softmax_probab = compute_softmax(img)
    return np.asarray(openmax_probab), np.asarray(softmax_probab)
# -------------------------------------------------------------------------------------------------------------------


def compute_openmax(mav_img, weibull, img, alpha=APLHA_RANK):
    ranked_list = img.argsort()
    alpha_weights = [((alpha + 1) - i) / float(alpha) for i in range(1, alpha + 1)]
    ranked_alpha = np.zeros(10)
    for i in range(len(alpha_weights)):
        ranked_alpha[ranked_list[9 - i]] = alpha_weights[i]

    imgarr = np.empty(1, dtype=object)
    imgarr[0] = img
    openmax_score, openmax_score_u = [], []
    for category in range(10):
        # category_weibull = query_weibull(labellist[categoryid], weibull_model, distance_type=distance_type)
        distance = compute_dist(mav_img[category], imgarr)

        # obtain w_score for the distance and compute probability of the distance
        # being unknown wrt to mean training vector and channel distances for
        # category and channel under consideration
        wscore = weibull[category].w_score(distance[0])

        modified_fc8_score = img[category] * (1 - wscore * ranked_alpha[category])
        openmax_score += [modified_fc8_score]
        openmax_score_u += [img[category] - modified_fc8_score]

    openmax_score = np.asarray(openmax_score)
    openmax_score_u = np.asarray(openmax_score_u)

    scores = []
    total_denominator = np.sum(np.exp(openmax_score)) + np.exp(np.sum(openmax_score_u))
    for i in range(len(img)):
        scores.append(np.exp(openmax_score[i]) / total_denominator)
    unknowns = np.exp(np.sum(openmax_score_u)) / total_denominator
    # print(np.sum(scores))
    # print(np.sum(scores) + unknowns)
    scores += [unknowns]
    # print(np.sum(scores))
    # print()
    return scores
# -------------------------------------------------------------------------------------------------------------------


def compute_softmax(last_fcl):
    softmax_scores = []
    for i in range(len(last_fcl)):
        softmax_scores.append(np.exp(last_fcl[i]) / np.sum(np.exp(last_fcl)))
    softmax_scores = np.asarray(softmax_scores)
    return softmax_scores
# -------------------------------------------------------------------------------------------------------------------


def main_one(num):
    openmax, softmax = main(num)
    print("openmax   %s" % openmax)
    print('sum = %s' % np.sum(openmax))
    print("softmax   %s" % softmax)
    print('sum = %s' % np.sum(softmax))
    print()
    return 0
# -------------------------------------------------------------------------------------------------------------------


def main_nok():  # not working atm, different files to main
    targets = np.load("preprocessing/train_output_targets.npy", allow_pickle=True)
    targets_nok = []
    for i in range(len(targets)):
        if not targets[i][1]:
            targets_nok.append(i)

    openmax, softmax = [], []
    for i in range(len(targets_nok)):
        om, sm = main(targets_nok[i])
        openmax.append(om)
        softmax.append(sm)
    openmax = np.asarray(openmax)
    softmax = np.asarray(softmax)
    print()
    return 0
# -------------------------------------------------------------------------------------------------------------------


def save_om_sm():
    last_layer = np.concatenate((np.load("preprocessing/c100_test_pred.npy", allow_pickle=True),
                                 np.load("preprocessing/c10_test_pred.npy", allow_pickle=True)), axis=0)
    mav = np.load("preprocessing/mav_c10_train.npy", allow_pickle=True)

    for tail_size in range(1, 31):
        weibull_model = weibull_tailfit(tail_size)
        for alpha in range(1, 9):
            openmax, softmax = [], []
            for example in range(len(last_layer)):
                img = last_layer[example]
                om = compute_openmax(mav, weibull_model, img, alpha)
                sm = compute_softmax(img)
                openmax.append(om)
                softmax.append(sm)
            openmax = np.asarray(openmax)
            softmax = np.asarray(softmax)
            np.save("testing/om_a%d_t%d.npy" % (alpha, tail_size), openmax)
            np.save("testing/sm_a%d_t%d.npy" % (alpha, tail_size), softmax)
# -------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    # main_one(IMG_INDEX)
    save_om_sm()