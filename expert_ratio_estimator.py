'''
## Train UU classifier ##
# Train the UU network on the demonstration samples to estimate the expert ratio
'''
import gc
import os
import sys

import IPython
import tensorflow as tf
import numpy as np
import argparse
import time

from utils.data_loader_v3 import DataGenerator
from utils.network import RewardNet
from utils.preprocess_trajs_offline import preprocess_trajs_offline
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from PU.KM2 import wrapper
from PU.KM3 import MPE_wrapper_RKME

# from PU.algorithm import *
# from PU.model_helper import *
# from PU.helper import *
from PU.estimator import *
# from measures import *

def get_args():
    args = argparse.ArgumentParser()

    args.add_argument("--env", type=str, help="Environment name", default="HalfCheetah-v2")
    args.add_argument("--ckpt_dir", type=str, help="Location to save the UU classifier checkpoints", default="./ckpt/HalfCheetah^alpha_0.1^policies_3^N_100")
    args.add_argument("--gpu", type=str, help="Choose gpu id", default="all")
    args.add_argument("--demo_path", type=str, help="Location of demonstrations", default="./demonstrations/mixture_3policies_ppo2.HalfCheetah.Reward_313.78_1563.28")
    args.add_argument("--estimator", type=str, help="choose mpe estimator", default="BBE", choices=["BBE", "KM"])
    args.add_argument("--alpha", type=float, default=0.1)
    args.add_argument("--N", type=int, default=100)
    args.add_argument("--restore_from_step", type=int, default=None, help="Checkpointed step to load and resume training from (if None, train from scratch)")
    args.add_argument("--n_train_epochs", type=int, default=3000)
    args.add_argument("--warm_epochs", type=int, default=500)
    args.add_argument("--val_trajs", type=int, default=1000, help="Number of trajectory pairs to generate before training to use as validation set")
    args.add_argument("--val_interval", type=int, default=100, help="Run validation every val_interval training steps and save checkpoint")
    args.add_argument("--log_interval", type=int, default=100)
    args.add_argument("--early_stopping_threshold", type=int, default=-1, help="Stop training after this number of validations without improvement")
    args.add_argument("--trn_batch_size", type=int, default=1024)
    args.add_argument("--val_batch_size", type=int, default=10240)
    args.add_argument("--learning_rate", type=float, default=5e-4)
    args.add_argument("--output_weight", action="store_true", default=False)
    args.add_argument("--preload", action="store_true", default=False)
    args.add_argument("--n_workers", type=int, default=16, help="For data loading and preprocessing")
    args.add_argument("--traj_length", type=int, default=50, help="We sample a random snippet of length traj_length from each demonstration sample to train on")
    args.add_argument("--noise", type=float, default=0.0, help="Noise level of generating data")

    # over-estimation switch
    args.add_argument("--over", help="turn to over estimation, i.e., 2 * beta / (1 + beta)", action="store_true", default=False)

    # PU hyperparameters
    args.add_argument("--delta", type=float, default=0.1)
    args.add_argument("--gamma", type=float, default=0.01)
    args.add_argument("--start_ratio", help="The ratio of the ratio for thrown data at the beginning",
                      type=float, default=0.1)
    args.add_argument("--end_ratio", help="The ratio of the ratio for thrown data in the end",
                      type=float, default=0.5)
    args.add_argument("--KM_sample_num", type=int, default=500)
    args.add_argument("--KM_version", type=int, default=3)
    return args.parse_args()


def rank_inputs_(probs, thrown_ratio, dataset="gamma-"):
    # probs < 0.5: gamma_positive
    u_size = probs.shape[0]
    keep_samples = np.ones_like(probs)

    if dataset == 'gamma-':
    # if dataset == 'gamma+':
        sorted_idx = np.argsort(probs)
    else:
        sorted_idx = np.argsort(-probs)

    keep_samples[sorted_idx[u_size - int(thrown_ratio*u_size):]] = 0
    return keep_samples

def D_comp_test(net, sess, data, args, name="positive"):
    batchsize = args.val_batch_size
    iters = data.shape[0] // batchsize + 1
    weight = []
    for i in range(iters):
        curdata = data[i * batchsize: (i + 1) * batchsize]
        weight.append(sess.run(net.reward_out_gamma_positive_probs, feed_dict={net.inputs_gamma_positive: curdata}))
    weight = np.concatenate(weight)
    acc = sum(weight >= 0.5) / weight.shape[0]
    if name == "negative":
        acc = 1 - acc
    return weight, acc

def calc_weight(net, sess, data, args, name="positive"):
    batchsize = args.val_batch_size
    iters = data.shape[0] // batchsize + 1
    weight = []
    print("Calculating weight for {} dataset ...".format(name))
    for i in tqdm(range(iters)):
        curdata = data[i * batchsize: (i + 1) * batchsize]
        weight.append(sess.run(net.reward_out_gamma_positive_probs, feed_dict={net.inputs_gamma_positive: curdata}))
    weight = np.concatenate(weight)
    acc = sum(weight >= 0.5) / weight.shape[0]
    if name == "negative":
        acc = 1 - acc
    print("Weight space: {} obs-acs shape: {} True accuracy: {:.4f}".format(weight.shape, data.shape, acc))
    return weight, acc

def discretize_array(gamma_positive_probs, gamma_negative_probs, num=100):
    array_range = np.linspace(min(gamma_positive_probs.min(), gamma_negative_probs.min()),
                              max(gamma_positive_probs.max(), gamma_negative_probs.max()), num+1)
    gamma_positive_idx = np.digitize(gamma_positive_probs, array_range, right=True)
    gamma_negative_idx = np.digitize(gamma_negative_probs, array_range, right=True)
    gamma_positive_discrete_probs = np.asarray([(array_range[i+1]+array_range[i])/2 for i in range(array_range.shape[0]-1)])
    gamma_negative_discrete_probs = np.asarray([(array_range[i+1]+array_range[i])/2 for i in range(array_range.shape[0]-1)])
    gamma_positive_weight = np.zeros_like(gamma_positive_discrete_probs)
    gamma_negative_weight = np.zeros_like(gamma_negative_discrete_probs)
    for idx in gamma_positive_idx:
        gamma_positive_weight[idx-1] += 1
    for idx in gamma_negative_idx:
        gamma_negative_weight[idx-1] += 1
    return gamma_positive_discrete_probs, gamma_positive_weight / gamma_positive_weight.sum(), \
           gamma_negative_discrete_probs, gamma_negative_weight / gamma_negative_weight.sum()

def train_val_process(args, train_data, val_data, data_positive, data_negative, true_alpha, trn_batch_size, val_batch_size,
                      sess, net, ckpt_dir, saver, early_stopping_threshold, beta=0, stage="train"):
    # Start training
    best_val_acc = 0.0
    # train_iters = 10000 // trn_batch_size
    val_iters = max(val_data[0].shape[0] // val_batch_size, 1)
    record_dir = "./uu_records"
    KM_sample_num = args.KM_sample_num
    os.makedirs(record_dir, exist_ok=True)
    # checkdir = os.path.join(record_dir, f"{ckpt_dir.split('/')[-2]}^{stage}_records-v2")
    checkdir = os.path.join(record_dir, f"{ckpt_dir.split('/')[-2]}^{stage}_records")
    if args.over:
        checkdir += "^over_estimation"
    os.makedirs(checkdir, exist_ok=True)
    epochs = args.warm_epochs
    if stage == "train":
        epochs = args.n_train_epochs
    gamma_negative_data = train_data[1][:, : -1]
    gamma_positive_data = train_data[0][:, : -1]
    gamma_negative_labels = train_data[1][:, -1]
    gamma_positive_labels = train_data[0][:, -1]
    thrown_ratios = np.linspace(args.start_ratio, args.end_ratio, epochs)
    for train_step in range(epochs):
        thrown_ratio = thrown_ratios[train_step]
        train_iters = max(train_data[0].shape[0] // trn_batch_size, 1)
        train_idxs_negative = np.arange(gamma_negative_labels.shape[0])
        train_idxs_positive = np.arange(gamma_positive_labels.shape[0])
        train_accs, val_accs, val_gamma_negative_probs, val_gamma_positive_probs = [], [], [], []
        np.random.shuffle(train_idxs_negative)
        np.random.shuffle(train_idxs_positive)
        train_losses = []
        cur_acc = []
        gamma_positive_probs, gamma_negative_probs = None, None
        curtime = time.time()
        for j in range(val_iters):
            val_gamma_negative = val_data[1][:, : -1][j * val_batch_size: (j + 1) * val_batch_size]
            val_gamma_positive = val_data[0][:, : -1][j * val_batch_size: (j + 1) * val_batch_size]
            gamma_positive_prob, gamma_negative_prob = sess.run([net.reward_out_gamma_positive_probs, net.reward_out_gamma_negative_probs],
                                           feed_dict={net.inputs_gamma_negative: val_gamma_negative,
                                                      net.inputs_gamma_positive: val_gamma_positive})
            if gamma_positive_probs is None:
                gamma_positive_probs = gamma_positive_prob
                gamma_negative_probs = gamma_negative_prob
            else:
                gamma_positive_probs = np.concatenate((gamma_positive_probs, gamma_positive_prob))
                gamma_negative_probs = np.concatenate((gamma_negative_probs, gamma_negative_prob))
            sys.stdout.write('\x1b[2K\rValidation Step: {:d}/{:d}'.format(j + 1, val_iters))
            sys.stdout.flush()

        # Estimate alpha, gamma_positiveer probs (to 1), more negative samples
        if stage == "train":
            if args.estimator == "BBE":
                if args.over:
                    # overestimate
                    _, beta, _ = BBE_estimator(gamma_positive_probs, gamma_negative_probs, np.ones_like(gamma_negative_probs), args.delta,
                                                         args.gamma)
                else:
                    # underestimate, mirror flip the probability
                    _, _, beta = BBE_estimator(1-gamma_negative_probs, 1-gamma_positive_probs, np.zeros_like(gamma_positive_probs), args.delta,
                                                         args.gamma)
            else:
                # underestimate
                if args.KM_version == 3:
                    gamma_positive_discrete_probs, gamma_positive_weight, \
                    gamma_negative_discrete_probs, gamma_negative_weight = discretize_array(gamma_positive_probs, gamma_negative_probs)
                    if args.over:
                        beta = MPE_wrapper_RKME(gamma_negative_discrete_probs, gamma_positive_discrete_probs,
                                                gamma_negative_discrete_probs.shape[0], gamma_positive_discrete_probs.shape[0],
                                                gamma_negative_weight, gamma_positive_weight)
                    else:
                        beta = MPE_wrapper_RKME(gamma_positive_discrete_probs, gamma_negative_discrete_probs,
                                                gamma_positive_discrete_probs.shape[0], gamma_negative_discrete_probs.shape[0],
                                                gamma_positive_weight, gamma_negative_weight)
                else:
                    gamma_negative_shuffle = np.random.permutation(len(gamma_negative_probs))
                    gamma_positive_shuffle = np.random.permutation(len(gamma_positive_probs))
                    if args.over:
                        beta = wrapper(gamma_negative_probs[gamma_negative_shuffle][:KM_sample_num],
                                       gamma_positive_probs[gamma_positive_shuffle][:KM_sample_num], KM_sample_num, KM_sample_num)
                    else:
                        beta = wrapper(gamma_positive_probs[gamma_positive_shuffle][:KM_sample_num],
                                       gamma_negative_probs[gamma_negative_shuffle][:KM_sample_num], KM_sample_num, KM_sample_num)

        # underestimation MPE
        if args.over:
            # overestimation
            original_alpha = (2 * beta) / (1 + beta)
        else:
            # under estimation
            original_alpha = (1 - beta) / (1 + beta)
        pos_thrown_ratio = (1 - original_alpha) ** 2 * thrown_ratio
        neg_thrown_ratio = original_alpha ** 2 * thrown_ratio
        val_gamma_positive_probs.append(gamma_positive_probs)
        val_gamma_negative_probs.append(gamma_negative_probs)
        if stage == "train":
            sys.stdout.write('\x1b[2K\r Alpha Estimated/True Alpha = {:.4f}/{:.4f} Elapsed Time = {:.2f}'.format(original_alpha, true_alpha, time.time() - curtime))
            sys.stdout.write('\n')
            sys.stdout.flush()
            keep_samples_gamma_negative = rank_inputs_(gamma_negative_probs, neg_thrown_ratio, 'gamma-')
            keep_samples_gamma_positive = rank_inputs_(gamma_positive_probs, pos_thrown_ratio, 'gamma+')

            kept_idx_gamma_negative = np.where(keep_samples_gamma_negative == 1)[0]
            kept_idx_gamma_positive = np.where(keep_samples_gamma_positive == 1)[0]
            val_acc = (sum(gamma_positive_probs[kept_idx_gamma_positive] > 0.5) + sum(gamma_negative_probs[kept_idx_gamma_negative] <= 0.5))\
                      * 100 / (gamma_positive_probs[kept_idx_gamma_positive].shape[0] + gamma_negative_probs[kept_idx_gamma_negative].shape[0])
        else:
            val_acc = (sum(gamma_positive_probs > 0.5) + sum(gamma_negative_probs <= 0.5))\
                      * 100 / (gamma_positive_probs.shape[0] + gamma_negative_probs.shape[0])
        sys.stdout.write('\x1b[2K\rEstimate alpha time: {:.2f} seconds'
                         '\nValidation Complete. '
                         '\t Val Acc = {:.2f}% '.format(time.time() - curtime, val_acc))
        sys.stdout.write('\n')
        sys.stdout.flush()

        val_accs.append(val_acc)

        # Throw samples for training
        curtime = time.time()
        kept_probs_gamma_negatives, kept_probs_gamma_positives, \
        kept_labels_gamma_negatives, kept_labels_gamma_positives, \
        thrown_probs_gamma_negatives, thrown_probs_gamma_positives, \
        thrown_labels_gamma_negatives, thrown_labels_gamma_positives, \
        thrown_idx_gamma_negative, thrown_idx_gamma_positive = [],[],[],[],[],[],[],[],[],[]
        cur_training_data_gamma_negative, cur_training_data_gamma_positive = [],[]
        # Throw data using estimated alpha
        if stage == "train":
            thrown_sample_iters = gamma_negative_data.shape[0] // val_batch_size
            for i in range(thrown_sample_iters):
                train_batch_idxs_positive = train_idxs_positive[i * val_batch_size: (i + 1) * val_batch_size]
                train_batch_idxs_negative = train_idxs_negative[i * val_batch_size: (i + 1) * val_batch_size]
                train_gamma_negative_data = gamma_negative_data[train_batch_idxs_negative]
                train_gamma_positive_data = gamma_positive_data[train_batch_idxs_positive]
                train_gamma_negative_labels = gamma_negative_labels[train_batch_idxs_negative]
                train_gamma_positive_labels = gamma_positive_labels[train_batch_idxs_positive]

                # net.create_train_step(gamma_positive_logits, gamma_negative_logits, trn_batch_size, opt)
                gamma_positive_probs, gamma_negative_probs = sess.run(
                    [net.reward_out_gamma_positive_probs, net.reward_out_gamma_negative_probs],
                    feed_dict={net.inputs_gamma_negative: train_gamma_negative_data,
                               net.inputs_gamma_positive: train_gamma_positive_data})

                keep_samples_gamma_negative = rank_inputs_(gamma_negative_probs, neg_thrown_ratio, 'gamma-')
                keep_samples_gamma_positive = rank_inputs_(gamma_positive_probs, pos_thrown_ratio, 'gamma+')

                kept_idx_gamma_negative = np.where(keep_samples_gamma_negative == 1)[0]
                kept_idx_gamma_positive = np.where(keep_samples_gamma_positive == 1)[0]

                thrown_idx_gamma_negative = np.where(keep_samples_gamma_negative == 0)[0]
                thrown_idx_gamma_positive = np.where(keep_samples_gamma_positive == 0)[0]
                if len(kept_idx_gamma_negative) < 1 or len(kept_idx_gamma_positive) < 1:
                    print(f"\x1b[2K\rIn Epoch {train_step}/{epochs}, no unlabeled data is chosen in {train_step:d}/{train_iters:d} "
                                     f"steps with alpha {beta:.4f}/{true_alpha:.4f}")
                    # continue
                kept_train_gamma_negative_data = train_gamma_negative_data[kept_idx_gamma_negative]
                kept_train_gamma_positive_data = train_gamma_positive_data[kept_idx_gamma_positive]
                kept_gamma_negative_probs = gamma_negative_probs[kept_idx_gamma_negative]
                kept_gamma_positive_probs = gamma_positive_probs[kept_idx_gamma_positive]

                thrown_probs_gamma_negative = gamma_negative_probs[thrown_idx_gamma_negative]
                thrown_probs_gamma_positive = gamma_positive_probs[thrown_idx_gamma_positive]

                thrown_probs_gamma_negatives.append(thrown_probs_gamma_negative)
                thrown_probs_gamma_positives.append(thrown_probs_gamma_positive)
                kept_probs_gamma_negatives.append(kept_gamma_negative_probs)
                kept_probs_gamma_positives.append(kept_gamma_positive_probs)

                thrown_labels_gamma_negatives.append(train_gamma_negative_labels[thrown_idx_gamma_negative])
                thrown_labels_gamma_positives.append(train_gamma_positive_labels[thrown_idx_gamma_positive])
                kept_labels_gamma_negatives.append(train_gamma_negative_labels[kept_idx_gamma_negative])
                kept_labels_gamma_positives.append(train_gamma_positive_labels[kept_idx_gamma_positive])

                # concat the remaining data with the thrown data from the other dataset
                cur_training_data_gamma_positive.append(np.concatenate(
                    (kept_train_gamma_positive_data, train_gamma_negative_data[thrown_idx_gamma_negative])))
                cur_training_data_gamma_negative.append(np.concatenate(
                    (kept_train_gamma_negative_data, train_gamma_positive_data[thrown_idx_gamma_positive])))

            cur_training_data_gamma_positive = np.concatenate(cur_training_data_gamma_positive)
            cur_training_data_gamma_negative = np.concatenate(cur_training_data_gamma_negative)
            train_idxs_positive = np.arange(len(cur_training_data_gamma_positive))
            train_idxs_negative = np.arange(len(cur_training_data_gamma_negative))
            np.random.shuffle(train_idxs_negative)
            np.random.shuffle(train_idxs_positive)
            train_iters = max(min(len(train_idxs_negative), len(train_idxs_positive)) // trn_batch_size, 1)

        # Train
        for i in range(train_iters):
            train_batch_idxs_negative = train_idxs_negative[i * trn_batch_size: (i + 1) * trn_batch_size]
            train_batch_idxs_positive = train_idxs_positive[i * trn_batch_size: (i + 1) * trn_batch_size]
            if stage == "train":
                train_gamma_negative_data = cur_training_data_gamma_negative[train_batch_idxs_negative]
                train_gamma_positive_data = cur_training_data_gamma_positive[train_batch_idxs_positive]
            else:
                train_gamma_negative_data = gamma_negative_data[train_batch_idxs_negative]
                train_gamma_positive_data = gamma_positive_data[train_batch_idxs_positive]

            train_loss, _, kept_gamma_positive_probs, kept_gamma_negative_probs = sess.run([net.loss, net.train_step,
                                                             net.reward_out_gamma_positive_probs, net.reward_out_gamma_negative_probs],
                                                            feed_dict={net.inputs_gamma_negative: train_gamma_negative_data,
                                                                       net.inputs_gamma_positive: train_gamma_positive_data})
            train_losses.append(train_loss)
            ave_train_loss = sum(train_losses) / len(train_losses)
            # train_acc = (sum(kept_gamma_positive_probs <= 0.5) + sum(kept_gamma_negative_probs > 0.5)) * 100 / (
            train_acc = (sum(kept_gamma_positive_probs > 0.5) + sum(kept_gamma_negative_probs <= 0.5)) * 100 / (
                        kept_gamma_positive_probs.shape[0] + kept_gamma_negative_probs.shape[0])
            cur_acc.append(train_acc)


            if (i + 1) % args.log_interval == 0 or (i + 1) == train_iters:
                time_interval = time.time() - curtime
                sys.stdout.write(f'\x1b[2K\rThrown data ratio gamma_negative: {len(thrown_idx_gamma_negative)}/{len(train_gamma_negative_data)}={len(thrown_idx_gamma_negative)/len(train_gamma_negative_data):.4f} '
                    f'gamma_positive: {len(thrown_idx_gamma_positive)}/{len(train_gamma_positive_data)}={len(thrown_idx_gamma_positive)/len(train_gamma_positive_data):.4f}'
                    f'\x1b[2K\rEpoch: {train_step + 1}/{epochs}\t{stage} Steps: {i + 1}/{train_iters}\t'
                    f'Train Acc = {train_acc:.2f}%\t'
                    f'Elapsed Time = {time_interval:.2f} s  '
                    f'Average Loss = {ave_train_loss:.4f}')
                # sys.stdout.write('\n')
                # print('\x1b[2K\rTrain Step: {:d}/{:d} \t Average Loss = {:.4f}'.format(train_step, n_train_epochs, ave_train_loss))
                sys.stdout.flush()

        train_accs.append(cur_acc)
        # Do validation
        sys.stdout.write('\n')
        sys.stdout.write('\n')
        sys.stdout.flush()

        curtime = time.time()
        weight_positive, pos_acc = D_comp_test(net, sess, data_positive, args, "positive")
        weight_negative, neg_acc = D_comp_test(net, sess, data_negative, args, "negative")
        all_acc = (neg_acc + pos_acc) / 2
        sys.stdout.write(
            f'Original Demo, positive {weight_positive.shape} acc/negative {weight_negative.shape} acc/all acc = {pos_acc:.4f}/{neg_acc:.4f}/{all_acc:.4f} '
            f'Elapsed Time = {time.time() - curtime:.2f}')
        sys.stdout.write('\n')
        sys.stdout.flush()

        target_file = "{}.alpha_{:.2f}.N_{:d}.gamma_{:.4f}" \
                      ".delta_{:.4f}.{}_records.epoch_{:d}.npz".format(args.env.split("-")[0],
                                                                           args.alpha, args.N,
                                                                           args.gamma, args.delta, stage, train_step+1)
        if stage == "train":
            thrown_probs_gamma_negatives = np.concatenate(thrown_probs_gamma_negatives)
            thrown_probs_gamma_positives = np.concatenate(thrown_probs_gamma_positives)
            kept_probs_gamma_negatives = np.concatenate(kept_probs_gamma_negatives)
            kept_probs_gamma_positives = np.concatenate(kept_probs_gamma_positives)

            kept_labels_gamma_negatives = np.concatenate(kept_labels_gamma_negatives)
            kept_labels_gamma_positives = np.concatenate(kept_labels_gamma_positives)
            thrown_labels_gamma_negatives = np.concatenate(thrown_labels_gamma_negatives)
            thrown_labels_gamma_positives = np.concatenate(thrown_labels_gamma_positives)

        if (train_step + 1) % 50 == 0:
            np.savez_compressed(os.path.join(checkdir, target_file),
                     train_acc=train_accs, val_acc=val_accs,
                     val_probs_gamma_negative=val_gamma_negative_probs, val_probs_gamma_positive=val_gamma_positive_probs, beta=beta,
                     pos_acc=pos_acc, neg_acc=neg_acc, all_acc=all_acc,
                     kept_probs_gamma_negative=kept_probs_gamma_negatives, kept_probs_gamma_positive=kept_probs_gamma_positives,
                     thrown_probs_gamma_negative=thrown_probs_gamma_negatives, thrown_probs_gamma_positive=thrown_probs_gamma_positives,
                     kept_labels_gamma_negative=kept_labels_gamma_negatives, kept_labels_gamma_positive=kept_labels_gamma_positives,
                     thrown_labels_gamma_negative=thrown_labels_gamma_negatives, thrown_labels_gamma_positive=thrown_labels_gamma_positives,
            )

        # if val_acc > best_val_acc:
        if (train_step + 1) % 50 == 0:
            # Save ckpt
            ckpt_path = os.path.join(ckpt_dir, 'Step_%05d.ckpt' % train_step)
            saver.save(sess, ckpt_path)
            best_val_acc = val_acc
    return net, beta

def train(args, train_data, val_data, data_positive, data_negative, true_alpha, ckpt_dir, restore_from_step, n_train_epochs, val_interval, early_stopping_threshold,
          trn_batch_size, val_batch_size, learning_rate, n_workers, traj_length):

    # Create network and training step op
    net = RewardNet(train_data[0].shape[1]-1, learning_rate, iterations=len(train_data[0])//trn_batch_size)

    # Create session and configure GPU options
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # Set up saver to save/load ckpts
    saver = tf.train.Saver(max_to_keep=100)
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)

    if args.preload:
        # Set up saver to save/load ckpts
        ckpt_path = tf.train.latest_checkpoint(os.path.join(args.ckpt_dir, "train"))

        print('Restoring from %s' % ckpt_path)
        saver.restore(sess, ckpt_path)
    else:
        if restore_from_step != None:
            ckpt_path = os.path.join(ckpt_dir, 'Step_%05d.ckpt' % (restore_from_step))
            print('Restoring from step %d' % (restore_from_step))
            saver.restore(sess, ckpt_path)
            start_step = restore_from_step + 1
        else:
            print('No checkpoint file found. Initialising...')
            init = tf.global_variables_initializer()
            sess.run(init)
            start_step = 1

    print('\nWarm starting... \n')

    cur_ckpt_dir = os.path.join(ckpt_dir, "warm_up")

    ckpt_path = tf.train.latest_checkpoint(cur_ckpt_dir)
    # ckpt_path = None

    if ckpt_path is not None:
        print('Restoring warm up from %s' % ckpt_path)
        saver.restore(sess, ckpt_path)
    else:
        net, alpha = train_val_process(args, train_data, val_data, data_positive, data_negative, true_alpha, trn_batch_size, val_batch_size,
                          sess, net, cur_ckpt_dir, saver, early_stopping_threshold, beta=0, stage="warm_up")

    cur_ckpt_dir = os.path.join(ckpt_dir, "train")

    net.init_learning_rate()

    print('\nTraining... \n')
    net, alpha = train_val_process(args, train_data, val_data, data_positive, data_negative, true_alpha, trn_batch_size, val_batch_size,
                      sess, net, cur_ckpt_dir, saver, early_stopping_threshold, beta=0, stage="train")
    return net, sess

def prepocess_train_val_split(demo_file, alpha, N):
    file = demo_file + f".alpha_{alpha:.2f}.N_{N}._split_demos.npz"

    print(f"Loading data from {file}")
    data = np.load(file)
    obs1 = data["obs1"]
    obs2 = data["obs2"]
    acs1 = data["acs1"]
    acs2 = data["acs2"]
    if "labels1" not in list(data.keys()):
        alpha = data['instance_alpha']
        num1 = round(obs1.shape[0] * (2 * alpha - alpha**2))
        num2 = round(obs1.shape[0] * ((1 - alpha)**2))
        labels1 = np.concatenate([np.ones(int(num1)), np.zeros(int(num2))])
        num1 = round(obs2.shape[0] * (alpha**2))
        num2 = round(obs2.shape[0] * (1 - alpha**2))
        labels2 = np.concatenate([np.ones(int(num1)), np.zeros(int(num2))])
        np.savez_compressed(file, **data, labels1=labels1, labels2=labels2)
        # np.savez_compressed(file, **variables_to_keep)
        data = np.load(file)
    labels1 = data["labels1"][:, np.newaxis]
    labels2 = data["labels2"][:, np.newaxis]

    obs_positive = data["obs_positive"]
    acs_positive = data["acs_positive"]
    obs_negative = data["obs_negative"]
    acs_negative = data["acs_negative"]
    data_negative = np.concatenate((obs_negative, acs_negative), 1)
    data_positive = np.concatenate((obs_positive, acs_positive), 1)

    instance_alpha = data["instance_alpha"]
    data1 = np.concatenate((obs1, acs1, labels1), 1)
    data2 = np.concatenate((obs2, acs2, labels2), 1)
    minlen = min(obs1.shape[0], obs2.shape[0])
    state_space = obs1.shape[1:]
    action_space = acs1.shape[1:]
    labels_space = [1]
    data1 = data1[: minlen, :]
    data2 = data2[: minlen, :]
    data = np.concatenate([data1, data2], 1)
    data_train, data_val = train_test_split(data, test_size=0.2, random_state=233)

    def recover_dim(data, state_space, action_space, label_space):
        obs1 = data[:, : state_space[0]]
        acs1 = data[:, state_space[0]: state_space[0] + action_space[0]]
        labels1 = data[:, state_space[0] + action_space[0]: state_space[0] + action_space[0] + label_space[0]]
        obs2 = data[:, state_space[0] + action_space[0] + label_space[0]:
                       state_space[0] + action_space[0] + label_space[0] + state_space[0]]
        acs2 = data[:, state_space[0] + action_space[0] + label_space[0] + state_space[0]:
                       state_space[0] + action_space[0] + label_space[0] + state_space[0] + action_space[0]]
        labels2 = data[:, state_space[0] + action_space[0] + label_space[0] + state_space[0] + action_space[0]:]
        if len(acs1.shape) == 1:
            acs1 = acs1[:, np.newaxis]
            acs2 = acs2[:, np.newaxis]
        return (np.concatenate((obs1, acs1, labels1), 1), np.concatenate((obs2, acs2, labels2), 1))

    data_train = recover_dim(data_train, state_space, action_space, labels_space)
    data_val = recover_dim(data_val, state_space, action_space, labels_space)
    print("Train size: {}, val size: {}".format(data_train[0].shape[0], data_val[0].shape[0]))
    return data_train, data_val, data_positive, data_negative, instance_alpha


def choose_gpu(args):
    import os, socket, pynvml
    pynvml.nvmlInit()
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(('8.8.8.8', 80))
    ip = s.getsockname()[0]
    s.close()
    if args.gpu == "all":
        memory_gpu = []
        masks = np.ones(pynvml.nvmlDeviceGetCount())
        for gpu_id, mask in enumerate(masks):
            if mask == -1:
                continue
            else:
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                memory_gpu.append(meminfo.free / 1024 / 1024)
        gpu1 = np.argmax(memory_gpu)
    else:
        gpu1 = args.gpu
    print("****************************Choosen GPU : {}****************************".format(gpu1))
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu1)
    os.environ['TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_IGNORE_PERFORMANCE'] = '1'
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

if __name__ == '__main__':
    args = get_args()
    choose_gpu(args)

    if args.noise != 0:
        args.ckpt_dir += f"^noise_{args.noise:.2f}"
    if "3policies" in args.demo_path:
        args.ckpt_dir += "^3policies"
    if "KM" in args.estimator:
        args.ckpt_dir += "^KM"
        data_train, data_val, data_positive, data_negative, true_alpha = prepocess_train_val_split(args.demo_path, args.alpha, args.N)
    # Train
    net, sess = train(args, data_train, data_val, data_positive, data_negative, true_alpha, args.ckpt_dir, args.restore_from_step, args.n_train_epochs, args.val_interval,
          args.early_stopping_threshold,
          args.trn_batch_size, args.val_batch_size, args.learning_rate, args.n_workers, args.traj_length)
    
    
    
    