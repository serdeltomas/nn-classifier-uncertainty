# -*- coding: utf-8 -*-

###################################################################################################
# Copyright (c) 2016 , Regents of the University of Colorado on behalf of the University          #
# of Colorado Colorado Springs.  All rights reserved.                                             #
#                                                                                                 #
# Redistribution and use in source and binary forms, with or without modification,                #
# are permitted provided that the following conditions are met:                                   #
#                                                                                                 #
# 1. Redistributions of source code must retain the above copyright notice, this                  #
# list of conditions and the following disclaimer.                                                #
#                                                                                                 #
# 2. Redistributions in binary form must reproduce the above copyright notice, this list          #
# of conditions and the following disclaimer in the documentation and/or other materials          #
# provided with the distribution.                                                                 #
#                                                                                                 #
# 3. Neither the name of the copyright holder nor the names of its contributors may be            #
# used to endorse or promote products derived from this software without specific prior           #
# written permission.                                                                             #
#                                                                                                 #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY             #
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF         #
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL          #
# THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,            #
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF     #
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)          #
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,           #
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS           #
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                                    #
#                                                                                                 #
# Author: Abhijit Bendale (abendale@vast.uccs.edu)                                                #
#                                                                                                 #
# If you use this code, please cite the following works                                           #
#                                                                                                 #
# A. Bendale, T. Boult “Towards Open Set Deep Networks” IEEE Conference on                        #
# Computer Vision and Pattern Recognition (CVPR), 2016                                            #
#                                                                                                 #
# Notice Related to using LibMR.                                                                  #
#                                                                                                 #
# If you use Meta-Recognition Library (LibMR), please note that there is a                        #
# difference license structure for it. The citation for using Meta-Recongition                    #
# library (LibMR) is as follows:                                                                  #
#                                                                                                 #
# Meta-Recognition: The Theory and Practice of Recognition Score Analysis                         #
# Walter J. Scheirer, Anderson Rocha, Ross J. Micheals, and Terrance E. Boult                     #
# IEEE T.PAMI, V. 33, Issue 8, August 2011, pages 1689 - 1695                                     #
#                                                                                                 #
# Meta recognition library is provided with this code for ease of use. However, the actual        #
# link to download latest version of LibMR code is: http://www.metarecognition.com/libmr-license/ #
###################################################################################################

import sklearn.metrics as slm
import numpy as np
import matplotlib.pyplot as pyplot

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