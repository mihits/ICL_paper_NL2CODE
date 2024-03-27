# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from concurrent.futures import process
import os
import csv
import json
import string
import numpy as np
import pickle as pkl
import math
import torch
import random

from collections import defaultdict
from functools import partial
from multiprocessing import Pool
from transformers import AutoTokenizer, AutoModel
from torch.nn import CosineSimilarity
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

class MetaICLData(object):

    def __init__(self, logger=None, gpt="gpt2-large", method="channel", 
                use_demonstrations=True, use_instruction=False, k=16,
                max_length=1024, max_length_per_example=256, do_tensorize=False, 
                tensorize_dir=None, n_process=None, n_gpu=None, local_rank=-1,
                add_newlines=False, n_prefix_tokens=0, prefix=True, 
                task_counts=None, prefix_token_ids=None, task=None, use_random_demo = False,use_similar_demo = False):

        self.logger = logger
        self.method = method
        self.use_demonstrations = use_demonstrations
        self.use_instruction = use_instruction
        self.k = k
        self.max_length = max_length
        self.max_length_per_example = max_length_per_example
        self.add_newlines = add_newlines
        self.n_prefix_tokens = n_prefix_tokens
        self.prefix = prefix
        self.task_counts = task_counts
        self.prefix_token_ids = prefix_token_ids
        self.task = task

        self.use_random_demo = use_random_demo
        self.use_similar_demo = use_similar_demo

        self.do_tensorize = do_tensorize
        self.tensorize_dir = tensorize_dir
        self.n_process = n_process
        self.n_gpu = n_gpu
        self.local_rank = local_rank

        self.tensorized_inputs = None
        self.metadata = None

        with open(os.path.join('config', 'causal_direction.json')) as f:
            causal_direction = json.load(f)

        with open(os.path.join('config', 'task_type.json')) as f:
            self.task_type = json.load(f)

        self.causal_direction = {}
        for k in causal_direction:
            self.causal_direction[k] = []
            for t in causal_direction[k]:
                self.causal_direction[k] += self.task_type[t]

        if gpt.startswith('gpt3'):
            self.tokenizer = AutoTokenizer.from_pretrained('gpt2-xl')
            # local_files_only=True
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(gpt)
            self.prefix_token_ids = []
            
            if self.n_prefix_tokens>0:
                if self.task_counts is None:
                    self.tokenizer = AutoTokenizer.from_pretrained(gpt, 
                        additional_special_tokens=[f'<t{i}>' for i in 
                            range(self.n_prefix_tokens)])
                    self.prefix_token_ids = self.tokenizer.additional_special_tokens_ids
                else:
                    self.tokenizer = AutoTokenizer.from_pretrained(gpt, 
                        additional_special_tokens=[f'<{task}-{i}>' 
                            for task in self.task_counts 
                            for i in range(self.n_prefix_tokens)])
                    self.prefix_token_ids = {} 
                    for i, task in enumerate(self.task_counts):
                        self.prefix_token_ids[task] = \
                            self.tokenizer.additional_special_tokens_ids[
                                i*self.n_prefix_tokens: (i+1)*self.n_prefix_tokens]
                print('prefix token ids: ', self.prefix_token_ids)
                ### example o/p: prefix token ids:  {'ag_news': [50257, 50258, 50259, 50260, 50261, 50262, 50263, 50264, 50265, 50266], 'emo': [50267, 50268, 50269, 50270, 50271, 50272, 50273, 50274, 50275, 50276]}
        

    def __len__(self):
        if self.tensorized_inputs is None:
            return 0
        return len(self.tensorized_inputs["input_ids"])

    def __str__(self):
        text = "[MetaICL Data]: method=%d, "
        if self.use_demonstrations:
            text += "%d demonstrations\n" % self.k
        else:
            text += "no demonstrations\n"
        if self.metadata is None:
            text += "Currently not containing any examples"
        else:
            text += "Currently containing %d examples with %d tensors to be fed in\n" % (len(self.metadata), len(self))
            text += "\n"
            text += self.print_tensorized_example(return_string=True)
        return ("="*50) + "\n" + text + "\n" + ("="*50)

    def get_dataloader(self, batch_size, is_training):
        inputs = self.tensorized_inputs
        for k, v in inputs.items():
            if type(v)==list:
                inputs[k] = torch.LongTensor(v)
        shape = inputs["input_ids"].shape
        self.logger.info(shape)
        for v in inputs.values():
            assert v.shape==shape
        if "labels" in inputs:
            dataset = TensorDataset(inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"], inputs["labels"])
        else:
            dataset = TensorDataset(inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"])
        if is_training:
            sampler=RandomSampler(dataset)
        else:
            sampler=SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
        return dataloader

    def evaluate(self, predictions, groundtruths, is_classification, return_all=False):
        # assert len(predictions)==len(self.metadata)
        accs = []
        precisions = defaultdict(list)
        recalls = defaultdict(list)
        for prediction, groundtruth in zip(predictions, groundtruths):
            prediction = prediction.strip()
            groundtruth = [gt.strip() for gt in groundtruth] if type(groundtruth)==list else groundtruth.strip()
            is_correct = prediction in groundtruth if type(groundtruth)==list else prediction==groundtruth
            accs.append(is_correct)
            if is_classification:
                recalls[groundtruth].append(is_correct)
                precisions[prediction].append(is_correct)

        if not return_all:
            accs = np.mean(accs)

        if not is_classification:
            return 0.0, accs

        f1s = []
        for key in recalls:
            precision = np.mean(precisions[key]) if key in precisions else 1.0
            recall = np.mean(recalls[key])
            if precision+recall==0:
                f1s.append(0)
            else:
                f1s.append(2*precision*recall / (precision+recall))
        print(np.mean(f1s))
        print(accs)

        return np.mean(f1s), accs

    def _prepro_each_datapoint(self, dp, is_first=True, is_training=False, for_demonstrations=False):
        dp = dp.copy()
        if self.method=="direct":
            method = "direct"
        elif self.method=="channel":
            method = "channel"
        elif self.method == "causal":
            if dp["task"] in self.causal_direction["x->y"]:
                method = "direct"
            elif dp["task"] in self.causal_direction["y->x"]:
                method = "channel"
            else:
                print("No such task in config.")
                raise NotImplementedError()
        elif self.method == "anti-causal":
            if dp["task"] in self.causal_direction["x->y"]:
                method = "channel"
            elif dp["task"] in self.causal_direction["y->x"]:
                method = "direct"
            else:
                print("No such task in config.")
                raise NotImplementedError()
        else:
            raise NotImplementedError()

        if self.add_newlines:  
            no_label = np.all([option=="" for option in dp["options"]])
            no_input = dp["input"]==""
            
            dp["input"] = dp["input"] + "\n"
            dp["input"] = "Input Instruction:\n" + dp["input"]
            dp["output"] = "Output Code:\n" + dp["output"]
            if method=="direct":
                
                if no_input:
                    dp["input"] = "\n\n" + dp["input"]
                else:
                    dp["input"] = "\n\n\n" + dp["input"]
                if not no_label:
                    dp["output"] = "\n" + dp["output"]
                    if "options" in dp:
                        dp["options"] = ["\n" + opt for opt in dp["options"]]
            elif method=="channel":
                if not is_first:
                    dp["output"] = "\n\n\n" + dp["output"]
                    if "options" in dp:
                        dp["options"] = ["\n\n\n" + opt for opt in dp["options"]]
                if not no_input:
                    if not no_label:
                        dp["input"] = "\n" + dp["input"]
        else:
            if not is_first:
                if method=="direct":
                    dp["input"] = " " + dp["input"]
                elif method=="channel":
                    dp["output"] = " " + dp["output"]
                    if "options" in dp:
                        dp["options"] = [" "+opt for opt in dp["options"]]

            if method=="direct":
                dp["output"] = " " + dp["output"]
                if "options" in dp:
                    dp["options"] = [" " + opt for opt in dp["options"]]
            elif method=="channel":
                dp["input"] = " " + dp["input"]

        

        if is_training or for_demonstrations:
            input_tokens = self.tokenizer(dp["input"])["input_ids"]
            output_tokens = self.tokenizer(dp["output"])["input_ids"]

            if "task" in dp:
                if (dp["task"].startswith("inst:piqa") or dp["task"].startswith("inst:yahoo_answers_topics")) and \
                        len(input_tokens)+len(output_tokens)+2>self.max_length_per_example:
                    input_tokens = input_tokens[:self.max_length_per_example // 2]
                    output_tokens = output_tokens[:self.max_length_per_example // 2 - 2]

                elif len(input_tokens)>=self.max_length_per_example - 2 - len(output_tokens):
                    if dp["task"].startswith("inst:") and len(input_tokens)<len(output_tokens):
                        output_tokens = output_tokens[:self.max_length_per_example - 2 - len(input_tokens)]
                    else:
                        input_tokens = input_tokens[:self.max_length_per_example - 2 - len(output_tokens)]

            assert len(input_tokens)+len(output_tokens)+2<=self.max_length_per_example, \
                (dp.get("task", None), len(input_tokens), len(output_tokens), self.max_length_per_example)

            if method=="direct":
                return input_tokens, output_tokens
            elif method=="channel":
                return output_tokens, input_tokens
            else:
                raise NotImplementedError()

        else:
            # assert len(dp["options"])>=2, dp
            # assert dp["output"] in dp["options"]
            # option_tokens = [self.tokenizer(option)["input_ids"] for option in dp["options"]]
            # option_length = np.max([len(option) for option in option_tokens])

            # if len(input_tokens)>=self.max_length_per_example - 2 - option_length:
            #     input_tokens = input_tokens[:self.max_length_per_example - 2 - option_length]

            # input_tokens = [input_tokens for _ in option_tokens] 
            # output_tokens = option_tokens
            # option_tokens = [dp["options"].index(dp["output"])]

            # if method=="direct":
            #     return input_tokens, output_tokens, option_tokens
            # elif method=="channel":
            #     return output_tokens, input_tokens, option_tokens
            # else:
            #     raise NotImplementedError()

            ###dp is one of the demos from saved_demos.json
            ###output_tokens = self.tokenizer(dp["output"])["input_ids"]
            
            output_tokens = self.tokenizer(" ")["input_ids"]  #since actual output doesnt matter for final inference
            ##since output token is null, append Output Code to end of input string
            dp["input"] += "Output Code:\n"
            input_tokens = self.tokenizer(dp["input"])["input_ids"]

            if len(input_tokens)>=self.max_length_per_example - 2:
                input_tokens = input_tokens[:self.max_length_per_example - 2]

            if method=="direct":
                return input_tokens, output_tokens
            elif method=="channel":
                return output_tokens, input_tokens
            else:
                raise NotImplementedError()

    
    # def retrive_most_sim(self, dp,_train_data_embed ):

    #     ip_tokens=tokenizer.tokenize(dp["input"])
    #     ip_tokens_ids=tokenizer.convert_tokens_to_ids(ip_tokens)
    #     model_op=sim_model(torch.tensor(ip_tokens_ids)[None,:])
    #     ip_embedding = model_op['last_hidden_state'][:,0,:]

    #     sim = CosineSimilarity(dim=-1)
    #     cosine = sim(ip_embedding, _train_data_embed )
    #     _, indices = cosine.topk(self.k)
    #     return indices

    # def embed_all(self, _train_data):
    #     ##retrive list of input keys' values
    #     pass


    def tensorize(self, _train_data, _test_data, instruction=None, options=None):
        ###_train_data = demos;   _test_data = curr_dev_data
        input_ids, attention_mask, token_type_ids = [], [], []
        metadata = []
        
        if self.use_instruction:
            if instruction is not None:
                inst_ids = self.tokenizer(instruction)["input_ids"]
            else:
                print("no instruction is given.")
                exit(1)

        if options is not None:  ###is none
            assert np.all([dp["output"] in options for dp in _train_data])
            for i, dp in enumerate(_test_data):
                assert "options" not in dp
                assert type(dp)==str
                _test_data[i] = {"input": dp, "options": options}

        train_data, test_data = [], []

        ###_train_data = list of dicts (all train data from which 4 random demos to be selected)
        
        if self.use_random_demo and self.use_similar_demo:
            print("Both random demo and similar demo selection enabled. Please select only one")
            exit(1)

        if self.use_random_demo:
            if type(_train_data[0])==dict:
                
                for _ in range(len(_test_data)):
                    demo = []
                    
                    #select 4 random demos from _train_data
                    demo = random.sample(_train_data, self.k)
                    train_data.append(demo)
                ### train_data contains n lists. (n= len of testdata);Each list contains a list of 4 demos.dimension = n*4

            else:
                print(_train_data)
                exit(1)

            demonstrations = []
            tasks = []
            for demo in train_data:
                assert len(demo)==self.k  ### demo is list of 4 demonstrations
                process_demo = []
                for i, dp in enumerate(demo):
                    input_, output_ = self._prepro_each_datapoint(
                        dp, is_first=i==0, for_demonstrations=True)
                    process_demo += input_ + output_
                demonstrations.append(process_demo)
                tasks.append(dp["task"])
            
            ### demonstrations -> length = n. each contains same tokenized 4 demos(combined together)


        elif self.use_similar_demo:
            if type(_train_data[0])==dict:
                
                for _ in range(len(_test_data)):
                    demo = []
                    
                    #select 4 random demos from _train_data
                    demo = random.sample(_train_data, self.k)
                    train_data.append(demo)


                #obtain embeddings (with similarity model) all of _train_data[this is list of dicts]

                for dp_idx, dp in enumerate(test_data):
                    pass
                    #select 4 most similar demos from _train_data

                    #for each dp, calc simlarity with embedding of _train_data {make function?}
                    #obtain indices of top_k most similar from above function
                    #sample from _train_data (list of dicts) using these indices
                    
                    


                ### train_data contains n lists. (n= len of testdata);Each list contains a list of 4 demos.dimension = n*4

            else:
                print(_train_data)
                exit(1)

            demonstrations = []
            tasks = []
            for demo in train_data:
                assert len(demo)==self.k  ### demo is list of 4 demonstrations
                process_demo = []
                for i, dp in enumerate(demo):
                    input_, output_ = self._prepro_each_datapoint(
                        dp, is_first=i==0, for_demonstrations=True)
                    process_demo += input_ + output_
                demonstrations.append(process_demo)
                tasks.append(dp["task"])
            
            ### demonstrations -> length = n. each contains same tokenized 4 demos(combined together)


        ### _train_data is a list of 4 dicts. each dict - represents a demo
        elif self.use_demonstrations:   
            if type(_train_data[0])==dict:
                for _ in range(len(_test_data)):
                    demo = []
                    for dp in _train_data:
                        assert type(dp)==dict, ("Each example should be a dictionary", dp)
                        assert "input" in dp and "output" in dp, ("Training example should contain input and output", dp)
                        demo.append(dp.copy())
                    train_data.append(demo)
                ### train_data contains n copies of _train_data i.e list of 4 demos(n= len of testdata);dimension = n*4
            
            elif type(_train_data[0])==list:
                if _test_data is not None:
                    assert len(_train_data) == len(_test_data)
                for _demo in _train_data:
                    demo = []
                    for dp in _demo:
                        assert type(dp)==dict, ("Each example should be a dictionary", dp)
                        assert "input" in dp and "output" in dp, ("Training example should contain input and output", dp)
                        demo.append(dp.copy())
                    train_data.append(demo)
            else:
                print(_train_data)
                exit(1)

            demonstrations = []
            tasks = []
            for demo in train_data:
                assert len(demo)==self.k  ### demo is list of 4 demonstrations
                process_demo = []
                for i, dp in enumerate(demo):
                    input_, output_ = self._prepro_each_datapoint(
                        dp, is_first=i==0, for_demonstrations=True)
                    process_demo += input_ + output_
                demonstrations.append(process_demo)
                tasks.append(dp["task"])
            
            ### demonstrations -> length = n. each contains same tokenized 4 demos(combined together)
        if _test_data is not None:
            for dp in _test_data:
                assert type(dp)==dict, ("Each example should be a dictionary", dp)
                assert "input" in dp and "options" in dp and type(dp["options"])==list, \
                    ("Test example should contain input and options in a list format", dp)
                if "output" not in dp:
                    dp["output"] = dp["options"][0] # randomly choose one (we don't need it anyways)
                test_data.append(dp.copy())

            ###test_data = _test_data

            # each datapoint: passage, question, options, output
            for dp_idx, dp in enumerate(test_data):
                inputs, outputs= self._prepro_each_datapoint(
                    dp, is_first=not self.use_demonstrations)
                task = dp["task"] if self.task is None else self.task


                # for inputs_, outputs_ in zip(inputs, outputs):
                #     if self.use_demonstrations:
                #         inputs_ = demonstrations[dp_idx] + inputs_
                #     if self.use_instruction:
                #         inputs_ = inst_ids + inputs_
                #     encoded = prepro_sentence_pair_single(
                #         inputs_, outputs_, self.max_length, self.n_prefix_tokens, 
                #         self.prefix_token_ids, self.prefix, self.use_demonstrations,
                #         task if self.task_counts is not None else None)
                #     input_ids.append(encoded[0])
                #     attention_mask.append(encoded[1])
                #     token_type_ids.append(encoded[2])
                #     inputs, outputs = self._prepro_each_datapoint(
                #     dp, is_first=not self.use_demonstrations)  ###is_first = true

                inputs = demonstrations[dp_idx] + inputs

                pad_tk = self.tokenizer.encode(self.tokenizer.pad_token)[0]
                encoded = prepro_sentence_pair_single(pad_tk,
                    inputs, outputs, self.max_length, self.n_prefix_tokens, 
                    self.prefix_token_ids, self.prefix, self.use_demonstrations,
                    task if self.task_counts is not None else None)  ### prefix = false

                input_ids.append(encoded[0])
                attention_mask.append(encoded[1])
                token_type_ids.append(encoded[2])
        else:
            for i, demo in enumerate(demonstrations):
                encoded = prepro_sentence_pair_single(
                    demo, [], self.max_length, self.n_prefix_tokens, 
                    self.prefix_token_ids, self.prefix, self.use_demonstrations,
                    tasks[i] if self.task_counts is not None else None)
                input_ids.append(encoded[0])
                attention_mask.append(encoded[1])
                token_type_ids.append(encoded[2])
                metadata.append({"indices": [i], "answer": 'None', 
                        "options": ['None'], "label": 0})

        self.tensorized_inputs = dict(input_ids=torch.LongTensor(input_ids),
                                      attention_mask=torch.LongTensor(attention_mask),
                                      token_type_ids=torch.LongTensor(token_type_ids))
        self.metadata = metadata

    

    def print_tensorized_example(self, return_string=False, print_all = False):
        assert self.tensorized_inputs is not None

        if not print_all:
            idx = 0
            text = "Checking the first example..."
            input_ids = self.tensorized_inputs["input_ids"][idx]
            token_type_ids = self.tensorized_inputs["token_type_ids"][idx]
            if type(input_ids)!=list:
                input_ids = input_ids.numpy().tolist()
            if type(token_type_ids)!=list:
                token_type_ids = token_type_ids.numpy().tolist()

            text += "\nInput:\n"
            text += self.tokenizer.decode(input_ids[:token_type_ids.index(1)])
            text += "\nOutput:\n"
            text += self.tokenizer.decode([_id for _id, _type_id in zip(input_ids, token_type_ids) if _type_id==1])

            if return_string:
                return text

            if self.local_rank<=0:
                self.logger.info(text)
        else:
            for idx in range(10000):
                text = ""
                ###text = "Checking the first example..."
                input_ids = self.tensorized_inputs["input_ids"][idx]
                token_type_ids = self.tensorized_inputs["token_type_ids"][idx]
                if type(input_ids)!=list:
                    input_ids = input_ids.numpy().tolist()
                if type(token_type_ids)!=list:
                    token_type_ids = token_type_ids.numpy().tolist()

                text += "\nInput:\n"
                text += self.tokenizer.decode(input_ids[:token_type_ids.index(1)])
                text += "\nOutput:\n"
                text += self.tokenizer.decode([_id for _id, _type_id in zip(input_ids, token_type_ids) if _type_id==1])

                with open('examples_step3_icl_formatted.txt', 'a') as file:
                    # Write data to the file
                    file.write(text)
                    file.write("\n\n")

    

def prepro_sentence_pair_single(pad_tk,ids1, ids2, max_length, n_prefix_tokens=0,
    prefix_token_ids=None, prefix=True, allow_truncation=True, task=None):  ###prefix = false, n_prefix_tokens =0

    #if bos_token_id is not None:
    #    ids1 = [bos_token_id] + ids1
    #if eos_token_id is not None:                       ###ids1 is input ; ids2 is output ;
    #    ids2 = ids2 + [eos_token_id]                   ###prefix_token_ids is dictionary mapping task to list? of token ids
    if len(ids1)+len(ids2)+n_prefix_tokens > max_length:
        if allow_truncation:
            if len(ids1) > len(ids2):
                ids1 = ids1[len(ids1)+len(ids2)-max_length+n_prefix_tokens:]
            else:
                ids2 = ids2[len(ids1)+len(ids2)-max_length+n_prefix_tokens:]
        else:
            if len(ids1) > len(ids2):
                ids1 = ids1[:max_length-n_prefix_tokens-len(ids2)]
            else:
                ids2 = ids2[:max_length-n_prefix_tokens-len(ids1)]
        assert len(ids1)+len(ids2)+n_prefix_tokens==max_length

    # if n_prefix_tokens > 0:
    #     if task is None:
    #         _prefix_token_ids = prefix_token_ids
    #     else:
    #         _prefix_token_ids = prefix_token_ids[task]

    #     if prefix:
    #         ids1 = _prefix_token_ids + ids1
    #     else:
    #         ids1 += ids2
    #         ids2 = _prefix_token_ids

    right_pad = False
    no_pad = False

    if right_pad:
        n_mask = max_length-len(ids1)-len(ids2)
        assert n_mask>=0, (max_length, len(ids1), len(ids2))
        input_ids = ids1+ids2+[0 for _ in range(n_mask)]
        attention_mask = [1 for _ in ids1+ids2] + [0 for _ in range(n_mask)]
        token_type_ids = [0 for _ in ids1] + [1 for _ in ids2] + [0 for _ in range(n_mask)]
    else:
        n_mask = max_length-len(ids1)-len(ids2)
        assert n_mask>=0, (max_length, len(ids1), len(ids2))
        input_ids = [pad_tk for _ in range(n_mask)]+ids1+ids2
        attention_mask = [0 for _ in range(n_mask)] + [1 for _ in ids1+ids2] 
        token_type_ids = [0 for _ in range(n_mask)] + [0 for _ in ids1] + [1 for _ in ids2] 
    
    if no_pad:
        input_ids = ids1+ids2
        attention_mask = [1 for _ in ids1+ids2] 
        token_type_ids = [0 for _ in ids1] + [1 for _ in ids2]
    return input_ids, attention_mask, token_type_ids



def load_data(task, split, k, seed=0, config_split=None, datasets=None,
              is_null=False, data_dir='data', full_train=False): 
    """load jsonl file and return a list of dictionaries(each dict represents one entry in dataset)"""

    if config_split is None:
        config_split = split

    if datasets is None:
        with open(os.path.join("config", task.strip()+".json"), "r") as f:
            config = json.load(f)
        datasets = config[config_split]  ###stores list of strings in tune.json where train is key (for tensorize&train.sh)

    data = []

    for dataset in datasets:
        if dataset == "humaneval":
            data_path = "data/humaneval/humaneval.jsonl"
        elif dataset == "mbbp":
            data_path = "data/mbbp/mbbp_dev.jsonl"
        else :
            print("invalid Dataset!\n")
            exit(1)
        # ###obtain name of data path where dataset jsonl file stored
        # # if split=='train':
        # #     if full_train:
        # #         data_path = os.path.join(data_dir, dataset, "{}_full_train.jsonl".format(dataset))
        # #     else:
        # #         data_path = os.path.join(data_dir, dataset,
        # #                 "{}_{}_{}_{}.jsonl".format(dataset, k, seed, split))
        # # elif split=='dev':
        # #     data_path = os.path.join(data_dir, dataset,
        # #                 "{}_{}_{}_{}.jsonl".format(dataset, k, seed, split))
        # # elif split=='test':
        # #     data_path = os.path.join(data_dir, dataset,
        # #                 "{}_test.jsonl".format(dataset))
        # # else:
        # #     print("choose split from [train, dev, test]")
        # #     exit(1)


        ###load jsonl file and return a list of dictionaries(each dict represents one entry in dataset)
        with open(data_path, "r") as f:
            for line in f:
                dp = json.loads(line)
                
                if is_null:
                    dp["input"] = "N/A"
                data.append(dp)

    return data