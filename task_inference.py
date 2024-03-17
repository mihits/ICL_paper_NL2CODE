# import sys
# sys.path.append('') 'give current path'
'''
write code for task inference using prompt
'''


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

from utils.load_inference_dataset import MetaICLData
from models.task_inference_model_pipeline import MetaICLModel
##from gpt3 import GPT3Model

from utils.load_inference_dataset import load_data

def main(logger, args):
    assert (args.dataset is not None and args.task is None) or (args.dataset is None and args.task is not None)

    if args.gpt.startswith("gpt3"):
        metaicl_model = GPT3Model(args.gpt[5:], args.api, logger)
        add_newlines = True
    else:
        ##add_newlines = not args.gpt.startswith("gpt2")
        
        add_newlines = True
        task_counts = None
        if args.prefix_embed_file is not None:
            model_dir = Path(args.prefix_embed_file).parent.absolute()
            if os.path.exists(os.path.join(model_dir, 'task2token.json')):
                with open(os.path.join(model_dir, 'task2token.json')) as f:
                    task_counts = json.load(f)

        metaicl_model = MetaICLModel(args.gpt, logger,
            soft_prefix=args.use_soft_prefix or args.use_soft_postfix,  #false for prior.sh
            n_tokens=args.n_prefix_tokens, prefix_embed_file=args.prefix_embed_file,
            task_counts=task_counts)
        
        if torch.cuda.is_available():
            metaicl_model.cuda()
        metaicl_model.eval()
    

    # setup hyperparams for data
    #max_length_per_example = int(args.max_length/args.k)  #length per demo
    max_length_per_example = args.max_length
    if args.use_demonstrations:       ### true for prior.sh
        max_length = min(max_length_per_example * args.k, args.max_length)
    else:
        max_length = max_length_per_example

    logger.info("batch_size=%d\tmax_length=%d\tmax_length_per_example=%d" % (
        args.test_batch_size, max_length, max_length_per_example))

    if args.use_soft_prefix or args.use_soft_postfix: ###both false
        metaicl_data = MetaICLData(logger, args.gpt, args.method,
            args.use_demonstrations, args.use_instruction, args.k, max_length, 
            max_length_per_example, add_newlines=add_newlines, 
            n_prefix_tokens=args.n_prefix_tokens, prefix=args.use_soft_prefix,
            task_counts=task_counts, prefix_token_ids=task_counts)
    else:
        metaicl_data = MetaICLData(logger, args.gpt, args.method,
            args.use_demonstrations, args.use_instruction, args.k,
            max_length, max_length_per_example, add_newlines=add_newlines)
    
    metaicl_data.tokenizer.padding_side = "left" 
    if metaicl_data.tokenizer.pad_token == None:
        metaicl_data.tokenizer.pad_token = metaicl_data.tokenizer.eos_token

    all_f1s = []
    all_accs = []
    errors = []
    all_scores = []
    all_dlls = []
    all_predictions = []
    seeds = args.seed.split(",")
    ###config_split = "unseen_domain_test" if args.unseen_domain_only else "test"
    config_split = "test"
    for seed in seeds:
        with open('saved_demos.json', 'r') as f:
            saved_demos = json.load(f)

        np.random.seed(int(seed))

        ###args.split default is test
        dev_data = load_data(args.task, args.split, args.k, seed=seed, config_split=config_split,
                             datasets=None if args.dataset is None else args.dataset.split(","), 
                             data_dir=args.data_dir, full_train=True)
        
        dev_counter = Counter()
        for dp in dev_data:
            dev_counter[dp["task"]] += 1
        for k, v in dev_counter.items():
            logger.info("[Dev] %s\t%d" % (k, v))
        logger.info("%s on %s ( %d dev)" % (args.method, args.task, len(dev_counter)))

        predictions_dict = {}

        for test_task in dev_counter:
            print("\n current test_task is ", test_task)
            curr_dev_data = [dp for dp in dev_data if dp["task"]==test_task]
            assert len(curr_dev_data)>0

            ### if length of curr_dev_data > test_size i.e1000 , then sample 1000 of them from curr_dev_data & discard rest
            if args.test_size < len(curr_dev_data) and args.split=="test":  ###args.test_size default 1000
                subsample_ids = np.random.choice(len(curr_dev_data), args.test_size, replace=False)
                curr_dev_data = np.array(curr_dev_data)[subsample_ids].tolist()

            config_file = "config/tasks/{}.json".format(test_task)
            assert os.path.exists(config_file), config_file
            with open(config_file, "r") as f:
                config = json.load(f)
            is_classification = config["task_type"] == "classification"
            is_multi_choice = config["task_type"] == "multi-choice"
            if is_classification:
                options = curr_dev_data[0]["options"]
                assert np.all([d["options"]==options for d in curr_dev_data])

        
            
            demonstrations = saved_demos[test_task]

            ### final inference after selecting demonstrations (done separately for every task,inside for loop)
            # f1, acc, pred, gt, nll, gt_label = run(test_task, metaicl_data, 
            #             metaicl_model, demonstrations, curr_dev_data, 
            #             is_classification) 
            preds = run(test_task, metaicl_data, 
                        metaicl_model, demonstrations, curr_dev_data, 
                        is_classification)
            
            predictions_dict[test_task] = preds


            # if save_path is not None and args.split=='train':  ###arg.split = test in prior.sh
            #     if not os.path.exists(save_path):
            #         os.makedirs(save_path)
            #     np.save(os.path.join(save_path, f'{args.split}-pred.npy'), pred)
            #     np.save(os.path.join(save_path, f'{args.split}-gt.npy'), gt_label)
            #     np.save(os.path.join(save_path, f'{args.split}-nll.npy'), nll)
            #     np.save(os.path.join(save_path, f'{args.split}-acc.npy'), acc)

            # all_predictions.append(pred)
            # logger.info("%s task (seed=%s): Macro-F1: %.1f, Accuracy: %.1f" % 
            #     (args.task, seed, 100*f1, 100*acc))
            # all_f1s.append(f1)
            # all_accs.append(acc)
        output_file_path = "predictions_" + args.gpt.split('/')[-1] + "_" + str(args.test_batch_size) + "_5.json"
        with open(output_file_path, 'w') as json_file:
            json.dump(predictions_dict, json_file, indent=2)
        
    


    # final_predictions = []
    # for p in np.transpose(all_predictions):
    #     v, c = np.unique(p, return_counts=True)
    #     final_predictions.append(v[np.argmax(c)])
    # final_f1, final_acc = metaicl_data.evaluate(final_predictions, gt, is_classification)
    # logger.info("%s over %d target tasks with majority vote: Macro-F1: %.1f, Accuracy: %.1f" % 
    #     (args.task, len(all_f1s) // len(seeds), 100*final_f1, 100*final_acc))

    # logger.info("%s over %d target tasks on average: Macro-F1: %.1f +- %.1f, Accuracy: %.1f +- %.1f" % 
    #     (args.task, len(all_f1s) // len(seeds), 100*np.mean(all_f1s), 100*np.std(all_f1s), 
    #     100*np.mean(all_accs), 100*np.std(all_accs)))

    # if len(errors)>0:
    #     logger.info("You had errors with datasets:", ",".join(errors))
    #     logger.info("Please see the error messages")



def run(task, metaicl_data, metaicl_model, train_data, dev_data,
        is_classification, return_all=False):

    if args.gpt.startswith("gpt3"):
        gpt3_dataloader, gpt3_metadata = metaicl_model.prepare_data(
            train_data if args.use_demonstrations else [],
            dev_data, args.method, batch_size=args.test_batch_size)
        losses, gpt3cache = metaicl_model.do_inference(gpt3_dataloader)	
        predictions, all_nlls, gt_labels, pred_labels = metaicl_model.do_predict(
            losses=losses, metadata=gpt3_metadata, return_nll=True)
    else:
      
        metaicl_data.tensorize(train_data, dev_data)
        metaicl_data.print_tensorized_example()

    
        preds = metaicl_model.perform_inference(metaicl_data, args.test_batch_size)

        return preds
        
       
        losses = metaicl_model.do_inference(metaicl_data, args.test_batch_size)
        assert len(losses)==len(metaicl_data)
        predictions, all_nlls, gt_labels, pred_labels = metaicl_model.do_predict(
            metaicl_data, losses=losses, return_nll=True)
    try:
        groundtruths = [dp["output"] for dp in dev_data]
        f1, acc = metaicl_data.evaluate(predictions, groundtruths, 
            is_classification, return_all)
        return f1, acc, predictions, groundtruths, all_nlls, gt_labels
    except:
        return all_nlls, gt_labels

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
    parser.add_argument("--gpt", type=str, default="gpt2-large")
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