import random
import numpy as np
import sys 

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.backends.cudnn as cudnn
# import torchvision
# import torchvision.transforms as transforms

# from baselines import *
# from data_helper import *
# from models import *

# from helper import *
# from model_helper import *

def DKW_bound(x,y,t,m,n,delta=0.1, gamma= 0.01):

    temp = np.sqrt(np.log(1/delta)/2/n) + np.sqrt(np.log(1/delta)/2/m)
    bound = temp*(1+gamma)/(y/n)

    estimate = t

    return estimate, t - bound, t + bound


def BBE_estimator(pdata_probs, udata_probs, udata_targets, delta=0.1, gamma=0.01):

    p_indices = np.argsort(pdata_probs)
    sorted_p_probs = pdata_probs[p_indices]
    u_indices = np.argsort(udata_probs)
    sorted_u_probs = udata_probs[u_indices]
    sorted_u_targets = udata_targets[u_indices]

    sorted_u_probs = sorted_u_probs[::-1]
    sorted_p_probs = sorted_p_probs[::-1]
    sorted_u_targets = sorted_u_targets[::-1]
    num = len(sorted_u_probs)

    estimate_arr = []

    upper_cfb = []
    lower_cfb = []            

    i = 0
    j = 0
    num_u_samples = 0

    while (i < num):
        start_interval = sorted_u_probs[i]
        k = i 
        if (i<num-1 and start_interval> sorted_u_probs[i+1]): 
            pass
        else: 
            i += 1
            continue
        # if (sorted_u_targets[i]==1):
        #     num_u_samples += 1

        while ( j<len(sorted_p_probs) and sorted_p_probs[j] >= start_interval):
            j+= 1

        if j>1 and i > 1:
            t = (i)*1.0*len(sorted_p_probs)/j/len(sorted_u_probs)
            estimate, lower , upper = DKW_bound(i, j, t, len(sorted_u_probs),
                                                len(sorted_p_probs), delta=delta, gamma=gamma)
            estimate_arr.append(estimate)
            upper_cfb.append( upper)
            lower_cfb.append( lower)
        i+=1

    if (len(upper_cfb) != 0): 
        idx = np.argmin(upper_cfb)
        mpe_estimate = estimate_arr[idx]
        mpe_estimate_lower = lower_cfb[idx]
        mpe_estimate_upper = upper_cfb[idx]

        # return mpe_estimate, lower_cfb, upper_cfb
        return mpe_estimate, mpe_estimate_lower, mpe_estimate_upper
    else:
        return 0.0, 0.0, 0.0





