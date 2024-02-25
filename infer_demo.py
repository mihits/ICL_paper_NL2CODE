# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse
import pickle as pkl
import random
import torch
import math
import json
import string
import logging
import time
import numpy as np
from datetime import datetime
from pathlib import Path

from tqdm import tqdm
from collections import Counter, defaultdict


from models.demo_inference_pipeline import DemoInferModel
from utils.load_demo_selection_dataset import load_data
from utils.load_demo_selection_dataset import DemoSelectionData



def main(logger, args):
    assert (args.dataset is not None and args.task is None) or (args.dataset is None and args.task is not None)

    if args.gpt.startswith("gpt3"):
        add_newlines = True
    else:
        add_newlines = not args.gpt.startswith("gpt2")
        
        task_counts = None
        if args.prefix_embed_file is not None:
            model_dir = Path(args.prefix_embed_file).parent.absolute()
            if os.path.exists(os.path.join(model_dir, 'task2token.json')):
                with open(os.path.join(model_dir, 'task2token.json')) as f:
                    task_counts = json.load(f)


    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # setup hyperparams for data
    max_length_per_example = 1024
    max_length = min(max_length_per_example * args.k, args.max_length)
    # if args.use_demonstrations:       ### true for prior.sh
    #     max_length = min(max_length_per_example * args.k, args.max_length)
    # else
    #     max_length = max_length_per_example

    logger.info("batch_size=%d\tmax_length=%d\tmax_length_per_example=%d" % (
        args.test_batch_size, max_length, max_length_per_example))

    seeds = args.seed.split(",")
    
    config_split = "test"
    for seed in seeds:
        np.random.seed(int(seed))

        ### data ...
        train_data = load_data(args.task, "train", args.k, 
            seed=seed, config_split=config_split,
            datasets=None if args.dataset is None else args.dataset.split(","),
            data_dir=args.data_dir, full_train=True)

        train_counter = Counter()
        for dp in train_data:
            train_counter[dp["task"]] += 1
        for k, v in train_counter.items():
            logger.info("[Train] %s\t%d" % (k, v))

        # ###args.split default is test
        # dev_data = load_data(args.task, args.split, args.k, seed=seed, config_split=config_split,
        #                      datasets=None if args.dataset is None else args.dataset.split(","), 
        #                      data_dir=args.data_dir, full_train=True)

        # dev_counter = Counter()
        
        # for dp in dev_data:
        #     dev_counter[dp["task"]] += 1
      
        # for k, v in dev_counter.items():
        #     logger.info("[Dev] %s\t%d" % (k, v))

        logger.info("%s on %s (%d train)" % (args.method, args.task, len(train_counter)))

        final_demo_dict = {}

        for test_task in train_counter:
            print("\n current test_task is ", test_task)
            # curr_dev_data = [dp for dp in dev_data if dp["task"]==test_task]
            # # assert len(curr_dev_data)>0

            # # ### if length of curr_dev_data > test_size i.e1000 , then sample 1000 of them from curr_dev_data & discard rest
            # # if args.test_size < len(curr_dev_data) and args.split=="test":  ###args.test_size default 1000
            # #     subsample_ids = np.random.choice(len(curr_dev_data), args.test_size, replace=False)
            # #     curr_dev_data = np.array(curr_dev_data)[subsample_ids].tolist()

            config_file = "config/tasks/{}.json".format(test_task)
            assert os.path.exists(config_file), config_file
            with open(config_file, "r") as f:
                config = json.load(f)
            is_classification = config["task_type"] == "classification"
            is_multi_choice = config["task_type"] == "multi-choice"
        
            _train_data = [dp for dp in train_data if dp["task"]==test_task]
            if args.train_size > 0:  ###args.train_size = 100
                subsample_ids = np.random.choice(len(_train_data), args.train_size, replace=False)
                curr_train_data = np.array(_train_data)[subsample_ids].tolist()
            else:
                curr_train_data = _train_data


            priors = set(args.prior)
            use_difficulty = len(set(["easiest", "hardest"]).intersection(priors))>0

            if len(args.prior) > 0:
                sorted_priors = sorted(args.prior)
                prior_text = '_'.join(sorted_priors)
                if use_difficulty:
                    prior_text += f"_diff={args.difficulty}"
            else:
                prior_text = 'uniform'

            if use_difficulty:   ###true for prior = easiest

                if args.difficulty == "concept_likelihood":
                    if args.train_size > 0:
                        concept_dir = os.path.join(args.concept_dir, 
                            f"{test_task}-train-{seed}")
                    else:
                        concept_dir = os.path.join(args.concept_dir, 
                            f"{test_task}-train")

                    if os.path.exists(concept_dir):
                        logger.info("loading saved concept likelihoods")
                        all_nll = np.load(os.path.join(concept_dir, f'{test_task}-nll.npy'))
                        gt_labels = np.load(os.path.join(concept_dir, f'{test_task}-gt.npy'))
                        
                    else:
                        assert args.prefix_embed_file is not None
                        model_dir = Path(args.prefix_embed_file).parent.absolute()
                        if os.path.exists(os.path.join(model_dir, 'task2token.json')):
                            with open(os.path.join(model_dir, 'task2token.json')) as f:
                                task_counts = json.load(f)
                        else:
                            task_counts = None
                        
                        logger.info("start running soft prefix model")
                        start_time = time.time()
                        concept_model = DemoInferModel(args.gpt, 
                            logger, args.out_dir, soft_prefix=True, 
                            n_tokens=args.n_prefix_tokens,
                            prefix_embed_file=args.prefix_embed_file, 
                            task_counts=task_counts)
                        if torch.cuda.is_available():
                            concept_model.cuda()
                        concept_model.eval()        

                        concept_data = DemoSelectionData(logger, args.gpt, 
                            args.method, False, args.use_instruction, args.k,
                            max_length, max_length_per_example, 
                            add_newlines=add_newlines, 
                            n_prefix_tokens=args.n_prefix_tokens,
                            prefix=False, task_counts=task_counts, 
                            prefix_token_ids=task_counts)

                        all_nll, gt_labels = run(test_task, 
                            concept_data, concept_model, None, curr_train_data, 
                            is_classification, None)

                        del concept_model
                        del concept_data
                        os.makedirs(concept_dir)
                        np.save(os.path.join(concept_dir, f'{test_task}-nll.npy'), all_nll)
                        np.save(os.path.join(concept_dir, f'{test_task}-gt.npy'), gt_labels)

                        logger.info(f"time use for computing {len(curr_train_data)} examples: {time.time()-start_time}")

                    opt_log_p = []
                    for _nll, l in zip(all_nll, gt_labels):
                        opt_log_p.append(-_nll[l]/args.concept_temperature)
                    opt_p = np.exp(opt_log_p)
                    difficulties = 1 - opt_p

                elif args.difficulty == "concept_calibrated":
                    
                    if args.prefix_embed_file is not None:
                        model_dir = Path(args.prefix_embed_file).parent.absolute()
                        if os.path.exists(os.path.join(model_dir, 'task2token.json')):
                            with open(os.path.join(model_dir, 'task2token.json')) as f:
                                task_counts = json.load(f)
                        else:
                            task_counts = None
                        
                    all_log_ps = []
                    difficulties = []

                    for task in task_counts:
                        if args.train_size > 0:
                            concept_dir = os.path.join(args.concept_dir, 
                                f"{task}-train-{seed}")
                        else:
                            concept_dir = os.path.join(args.concept_dir, 
                                f"{task}-train")

                        if os.path.exists(concept_dir):
                            logger.info("loading saved concept likelihoods")
                            all_nll = np.load(os.path.join(concept_dir, f'{task}-nll.npy'))
                            gt_labels = np.load(os.path.join(concept_dir, f'{task}-gt.npy'))
                            
                        else:
                            assert args.prefix_embed_file is not None
                            logger.info("start running soft prefix model")
                            start_time = time.time()
                            concept_model = DemoInferModel(args.gpt, 
                                logger, args.out_dir, soft_prefix=True, 
                                n_tokens=args.n_prefix_tokens,
                                prefix_embed_file=args.prefix_embed_file, 
                                task_counts=task_counts)
                            if torch.cuda.is_available():
                                concept_model.cuda()
                            concept_model.eval()        

                            concept_data = DemoSelectionData(logger, args.gpt, 
                                args.method, False, args.use_instruction, args.k,
                                max_length, max_length_per_example, 
                                add_newlines=add_newlines, 
                                n_prefix_tokens=args.n_prefix_tokens,
                                prefix=False, task_counts=task_counts, 
                                prefix_token_ids=task_counts, task=task)

                            all_nll, gt_labels = run(test_task, concept_data, 
                                concept_model, None, curr_train_data, is_classification, 
                                None)

                            del concept_model
                            del concept_data
                            os.makedirs(concept_dir)
                            np.save(os.path.join(concept_dir, f'{task}-nll.npy'), all_nll)
                            np.save(os.path.join(concept_dir, f'{task}-gt.npy'), gt_labels)

                            logger.info(f"time use for computing {len(curr_train_data)} examples: {time.time()-start_time}")

                        log_p = []

                        ### find out dimension of all_nll for classif & nl2code
                        ### 
                        for _nll, l in zip(all_nll, gt_labels):
                            log_p.append(-_nll[l]/args.concept_temperature)

                        # for _nll in all_nll:
                        #     log_p.append(-_nll/args.concept_temperature)


                        if task == test_task:
                            opt_log_p = log_p
                        all_log_ps.append(log_p)
                    
                    z=0
                    for log_p in all_log_ps:
                        z += np.exp(log_p)
                    calibrated_p = np.exp(opt_log_p - np.log(z))
                    difficulties = 1-calibrated_p
                    
                else:
                    print(f"{args.difficulty} is not defined.")
                    exit(1)

                difficulties = np.array(difficulties)
                assert len(difficulties) == len(curr_train_data)
                ##print(difficulties)

                sorted_diff = np.sort(difficulties)
                min_diff = sorted_diff[0]
                logger.info(f"min difficulty: {min_diff}")
                max_diff = sorted_diff[-1]
                
                logger.info(f"max difficulty: {max_diff}")
                logger.info(f"average difficulty: {np.mean(difficulties)}")
                



            if "hardest" in args.prior or "easiest" in args.prior:
                if "balanced" in args.prior and is_classification:
                    pass
                    '''
                        all_labels = np.array([d["output"] for d in curr_train_data])
                        all_ids = np.arange(len(curr_train_data))
                        _k = math.ceil(args.k/len(options))
                        top_ids = []
                        top_diff = []
                        for c in options:
                            curr_ids = all_ids[all_labels == c]
                            sorted_ids = curr_ids[np.argsort(difficulties[all_labels == c])]
                            if "hardest" in args.prior:
                                top_ids += list(sorted_ids[-_k:])
                                top_diff += list(sorted_diff[-_k:])
                            else:
                                top_ids += list(sorted_ids[:_k])
                                top_diff += list(sorted_diff[:_k])
                        top_ids = np.array(top_ids)
                        if "hardest" in args.prior:
                            demo_ids = top_ids[np.argsort(top_diff)[-args.k:]]
                        else:
                            demo_ids = top_ids[np.argsort(top_diff)[:args.k]]
                    '''
                else:
                    sorted_ids = np.argsort(difficulties)
                    if "hardest" in args.prior:
                        demo_ids = sorted_ids[-args.k:]
                    else:
                        demo_ids = sorted_ids[:args.k]
                
                
                if args.reorder:
                    demo_ids_perm = permutation(list(demo_ids))
                    demos_perm = [[curr_train_data[i] for i in _demo_ids] 
                        for _demo_ids in demo_ids_perm]
                    save_dir = os.path.join(args.concept_dir, "reorder")
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)

                    if args.difficulty == "concept_likelihood":
                        curr_nll_path = os.path.join(save_dir, 
                            f"{test_task}-nll-demos_perm-{seed}.npy")
                        demo_ids_path = os.path.join(save_dir, 
                            f"{test_task}-reordered_demo_ids-{seed}.npy")

                        if os.path.exists(demo_ids_path):
                            demo_ids = np.load(demo_ids_path)

                        if os.path.exists(curr_nll_path):
                            logger.info("loading saved demo nlls")
                            all_nll = np.load(curr_nll_path)
                        else:
                            logger.info("start running soft prefix model")
                            start_time = time.time()
                            concept_model = DemoInferModel(args.gpt, 
                                logger, args.out_dir, soft_prefix=True, 
                                n_tokens=args.n_prefix_tokens,
                                prefix_embed_file=args.prefix_embed_file, 
                                task_counts=task_counts)
                            if torch.cuda.is_available():
                                concept_model.cuda()
                            concept_model.eval()

                            concept_data = DemoSelectionData(logger, args.gpt, 
                                args.method, True, args.use_instruction, args.k,
                                max_length, max_length_per_example, 
                                add_newlines=add_newlines, 
                                n_prefix_tokens=args.n_prefix_tokens,
                                prefix=False, task_counts=task_counts, 
                                prefix_token_ids=task_counts)

                            all_nll, gt_labels = run(test_task, 
                                concept_data, concept_model, demos_perm, 
                                None, is_classification, None)

                            del concept_model
                            del concept_data

                            logger.info(f"time use for computing {len(demos_perm)} examples: {time.time()-start_time}")
                            np.save(curr_nll_path, all_nll)

                        opt_log_p = []
                        for _nll in all_nll:
                            opt_log_p.append(-_nll[0]/args.concept_temperature)
                        demo_ids = demo_ids_perm[np.argmax(opt_log_p)]
                        np.save(demo_ids_path, demo_ids)

                    elif args.difficulty == "concept_calibrated":
                        demo_ids_path = os.path.join(save_dir, 
                            f"{test_task}-reordered_demo_ids-cali-{seed}.npy")
                        if os.path.exists(demo_ids_path):
                            demo_ids = np.load(demo_ids_path)

                        if not os.path.exists(os.path.join(args.concept_dir, 
                            "reorder")):
                            os.makedirs(os.path.join(args.concept_dir, "reorder"))

                        if not os.path.exists(os.path.join(args.concept_dir, 
                            "prefix")):
                            os.makedirs(os.path.join(args.concept_dir, "prefix"))

                        all_log_ps = []
                        all_prefix_ps = []
                        for task in task_counts:
                            curr_nll_path = os.path.join(args.concept_dir, 
                                "reorder", f"{task}-nll-demos_perm-{seed}.npy")

                            curr_prefix_p_path = os.path.join(args.concept_dir, 
                                "prefix", f"{task}-p-{seed}.npy")

                            concept_model = DemoInferModel(args.gpt, 
                                logger, args.out_dir, soft_prefix=True, 
                                n_tokens=args.n_prefix_tokens,
                                prefix_embed_file=args.prefix_embed_file, 
                                task_counts=task_counts)
                            if torch.cuda.is_available():
                                concept_model.cuda()
                            concept_model.eval()

                            if os.path.exists(curr_nll_path):
                                logger.info("loading saved demo nlls")
                                all_nll = np.load(curr_nll_path)
                            else:
                                start_time = time.time()
                                concept_data = DemoSelectionData(logger, args.gpt, 
                                    args.method, True, args.use_instruction, args.k,
                                    max_length, max_length_per_example, 
                                    add_newlines=add_newlines, 
                                    n_prefix_tokens=args.n_prefix_tokens,
                                    prefix=False, task_counts=task_counts, 
                                    prefix_token_ids=task_counts, task=task)

                                all_nll, _ = run(test_task, 
                                    concept_data, concept_model, demos_perm, 
                                    None, is_classification, None)

                                del concept_data

                                logger.info(f"time use for computing {len(demos_perm)} examples: {time.time()-start_time}")
                                np.save(curr_nll_path, all_nll)

                            del concept_model

                            log_p = []
                            for _nll in all_nll:
                                log_p.append(-_nll[0]/args.concept_temperature)

                            if task == test_task:
                                opt_log_p = log_p
                            all_log_ps.append(log_p)
                        
                        z=0
                        for log_p in all_log_ps:
                            z += np.exp(log_p)
                        calibrated_p = np.exp(opt_log_p - np.log(z))

                        ###obtained index of selected demonstrations in curr_train_data
                        demo_ids = demo_ids_perm[np.argmax(calibrated_p)]
                        np.save(demo_ids_path, demo_ids)
            else:   
                demo_ids = np.random.choice(len(curr_train_data), 
                    args.test_size*args.k)


            demonstrations = []
            for i in demo_ids:
                demonstrations.append(curr_train_data[i])
                
            if len(demo_ids) != args.k:
                demo_ids = np.reshape(demo_ids, (args.test_size, args.k))

            if len(demonstrations) == args.k:
                save_path = None
            else:
                demonstrations = np.reshape(demonstrations, 
                    (args.test_size, args.k))
            
            
            final_demo_dict[test_task] = demonstrations

        with open('saved_demos.json', 'w') as fp:
                json.dump(final_demo_dict, fp, indent = 2)




def permutation(lst):
    ### returns a list containing all the permutations of the input list.
    if len(lst) == 0:
        return []
    if len(lst) == 1:
        return [lst]
 
    l = [] 
    for i in range(len(lst)):
       m = lst[i]
       remLst = lst[:i] + lst[i+1:]
       for p in permutation(remLst):
           l.append([m] + p)
    return l

def run(task, metaicl_data, metaicl_model, train_data, dev_data,
        is_classification, save_path, return_all=False):
    
    metaicl_data.tensorize(train_data, dev_data)   ###train_data = [], dev_data = curr_train_data
    metaicl_data.print_tensorized_example()

    losses = metaicl_model.do_inference(metaicl_data, args.test_batch_size)
    assert len(losses)==len(metaicl_data)
    # predictions, all_nlls, gt_labels, pred_labels = metaicl_model.do_predict(
    #     metaicl_data, losses=losses, return_nll=True)
    # return all_nlls, gt_labels

    all_nlls = metaicl_model.do_predict(metaicl_data, losses=losses, return_nll=True)
    return all_nlls

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--use_demonstrations", default=False, action="store_true")
    parser.add_argument("--use_soft_prefix", default=False, action="store_true")
    parser.add_argument("--use_soft_postfix", default=False, action="store_true")
    parser.add_argument("--n_prefix_tokens", type=int, default=10)
    parser.add_argument("--max_length", type=int, default=1024)

    parser.add_argument("--prior", type=str, nargs='+', default=[], 
        choices=["most_similar", "easiest", "hardest"])
    parser.add_argument("--difficulty", type=str, default="length", 
        choices=["concept_likelihood", "concept_calibrated"])
    parser.add_argument("--reorder", default=False, action="store_true")

    parser.add_argument("--log_dir", default='logs', type=str)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--load_dir", default=None, type=str)
    parser.add_argument("--concept_dir", default=None, type=str)
    parser.add_argument("--prefix_embed_file", default=None, type=str)

    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--seed", type=str, default="100,13,21,42,87")

    parser.add_argument("--test_batch_size", type=int, default=32)
    parser.add_argument("--global_step", type=str, default=None)
    parser.add_argument("--use_random_english_words", default=False, action="store_true")
    parser.add_argument("--use_random_label", default=False, action="store_true")

    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--method", type=str, default="direct", 
        choices=["direct", "channel"])
    parser.add_argument("--gpt", type=str, default="gpt2-large", 
        choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl",
                "gpt3-ada", "gpt3-babbage", "gpt3-curie", "gpt3-davinci", 
                "gpt3-text-ada-001", "gpt3-text-babbage-001", "gpt3-text-curie-001", 
                "gpt3-text-davinci-001", "gpt3-text-davinci-002", 
                "gpt3-code-davinci-002", "gpt3-text-davinci-003"])
    parser.add_argument("--api", type=str, default=None)

    parser.add_argument("--test_size", type=int, default=1000)
    parser.add_argument("--train_size", type=int, default=100)
    parser.add_argument("--embedding_dir", type=str, default='embedding')
    parser.add_argument("--embedding_model", type=str, default='all-mpnet-base-v2', 
        choices=['all-mpnet-base-v2'])
    parser.add_argument("--similarity_temperature", type=float, default=1.0)
    parser.add_argument("--concept_temperature", type=float, default=10.0)
    parser.add_argument("--use_instruction", default=False)

    args = parser.parse_args()
    
    os.makedirs(args.log_dir, exist_ok=True)
    log_file = os.path.join(args.log_dir, datetime.fromtimestamp(time.time()).isoformat())
    handlers = [logging.StreamHandler(), logging.FileHandler(log_file)]

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        handlers=handlers)
    logger = logging.getLogger(__name__)
    logger.info(args)

    main(logger, args)






    