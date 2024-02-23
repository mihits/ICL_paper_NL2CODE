'''
Dataloader with shifted tokens and set predefined set of tokens
'''
from torch.utils.data import DataLoader
from concurrent.futures import process
import os
import csv
import json
import string
import numpy as np
import pickle as pkl
import math
import torch

from collections import defaultdict
from functools import partial
from multiprocessing import Pool
from transformers import AutoTokenizer

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
class PromptLearningDataset():
    def __init__(self, logger=None, gpt="gpt2-large", method="channel", 
                use_demonstrations=False, use_instruction=False, k=16,
                max_length=1024, max_length_per_example=256,
                tensorize_dir=None, n_process=None, n_gpu=None, local_rank=-1,
                add_newlines=False, n_prefix_tokens=0, prefix=True, 
                task_counts=None, prefix_token_ids=None, task=None):

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

    
    def print_tensorized_example(self, return_string=False):
        assert self.tensorized_inputs is not None

        # idx = 0
        # text = "Checking the first example..."
        # input_ids = self.tensorized_inputs["input_ids"][idx]
        # token_type_ids = self.tensorized_inputs["token_type_ids"][idx]
        # if type(input_ids)!=list:
        #     input_ids = input_ids.numpy().tolist()
        # if type(token_type_ids)!=list:
        #     token_type_ids = token_type_ids.numpy().tolist()

        # text += "\nInput:\n"
        # text += self.tokenizer.decode(input_ids[:token_type_ids.index(1)])
        # text += "\nOutput:\n"
        # text += self.tokenizer.decode([_id for _id, _type_id in zip(input_ids, token_type_ids) if _type_id==1])

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

            with open('step1_examples.txt', 'a') as file:
                # Write data to the file
                file.write(text)
                file.write("\n\n")

        if return_string:
            return text

        if self.local_rank<=0:
            self.logger.info(text)

        
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

        if self.add_newlines:  ###false
            no_label = np.all([option=="" for option in dp["options"]])
            no_input = dp["input"]==""
            
            if method=="direct":
                if not is_first:
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
            if not is_first:  ##is_first = true
                if method=="direct":
                    dp["input"] = " " + dp["input"]
                elif method=="channel":
                    dp["output"] = " " + dp["output"]
                    if "options" in dp:
                        dp["options"] = [" "+opt for opt in dp["options"]]

            if method=="direct":
                ###dp["output"] = " " + dp["output"]
                if "options" in dp:
                    dp["options"] = [" " + opt for opt in dp["options"]]
            elif method=="channel":
                dp["input"] = " " + dp["input"]

        input_tokens = self.tokenizer(dp["input"])["input_ids"]

        if is_training or for_demonstrations:  ###is_training = true
            output_tokens = self.tokenizer(dp["output"])["input_ids"]

            if "task" in dp:
                if len(input_tokens)>=self.max_length_per_example - 2 - len(output_tokens):
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



    def _tensorize_for_training(self, train_data):
        for dp in train_data:
            assert type(dp)==dict, ("Each example should be a dictionary", dp)
            assert "input" in dp and "output" in dp, ("Training example should contain input and output", dp)

        # each datapoint: passage, question, options, output
    
        input_ids, attention_mask, token_type_ids = [], [], []

        for dp in train_data:
            inputs, outputs = self._prepro_each_datapoint(
                dp, is_first=True, is_training=True)
            task = dp["task"] if self.task is None else self.task

            encoded = append_prefix(
                inputs, outputs, self.max_length, self.n_prefix_tokens,
                self.prefix_token_ids, allow_truncation=True, 
                task=task if self.task_counts is not None else None)

            input_ids.append(encoded[0])
            attention_mask.append(encoded[1])
            token_type_ids.append(encoded[2])

        return dict(input_ids=torch.LongTensor(input_ids),
                    attention_mask=torch.LongTensor(attention_mask),
                    token_type_ids=torch.LongTensor(token_type_ids))



    def tensorize_for_training(self, train_data, keyword, seed, use_random_english_words=False):
        assert self.tensorize_dir is not None

        if not os.path.exists(self.tensorize_dir):
            os.makedirs(self.tensorize_dir)

        method_name = self.method 
        k_name = len(train_data)
        length_name = "%d-%d" % (self.n_prefix_tokens, self.max_length)
        postfix = ""

        tensorize_path = os.path.join(self.tensorize_dir,
                                    "{}_{}_k={}_seed={}_length={}{}-rank=%d.pkl".format(
                                        keyword, method_name, k_name, seed, length_name,
                                        postfix))

        if self.local_rank==-1:
            self.logger.info(tensorize_path)
        else:
            self.logger.info(tensorize_path % self.local_rank)
        all_tensorize_paths = [tensorize_path % i for i in range(self.n_gpu)]


        assert self.local_rank==-1
        if any([os.path.exists(_path) for _path in all_tensorize_paths]):
            self.logger.info("tensorize file already exists...")
            return

        unique_task_names = set([dp["task"] for dp in train_data])
        sharded_inputs = []
        if self.use_demonstrations or (len(unique_task_names)>200 and len(train_data)>=1638400):
            tot = 0
            for i, curr_train_task in enumerate(unique_task_names):
                curr_train_data = [dp for dp in train_data if dp["task"]==curr_train_task]
                tot += len(curr_train_data)
                if self.use_demonstrations and len(unique_task_names)>200 and len(train_data)>=1638400:
                    # data is too huge; sampling 10% of the data
                    self.logger.info("Sampling training data from %d to %d", len(curr_train_data), len(curr_train_data)//10)
                    indices = np.random.permutation(range(len(curr_train_data)))[:len(curr_train_data)//10]
                    curr_train_data = [curr_train_data[i] for i in indices]
                elif len(unique_task_names)>200 and len(train_data)>=1638400:
                    # data is too huge; sampling 50% of the data
                    self.logger.info("Sampling training data from %d to %d", len(curr_train_data), len(curr_train_data)//2)
                    indices = np.random.permutation(range(len(curr_train_data)))[:len(curr_train_data)//2]
                    curr_train_data = [curr_train_data[i] for i in indices]
                sharded_inputs.append(curr_train_data)
            assert len(train_data)==tot
        else:
            n_per_shard = math.ceil(len(train_data) / self.n_process)
            for i in range(self.n_process):
                sharded_inputs.append(train_data[i*n_per_shard:(i+1)*n_per_shard])

        inputs = {"input_ids": [], "attention_mask": [], "token_type_ids": []}
        ## _tensorize_for_training = self._tensorize_for_training_with_random_english_words \
        ##     if use_random_english_words else self._tensorize_for_training

        _tensorize_for_training = self._tensorize_for_training
        if self.n_process==1:
            for in_ in sharded_inputs:
                out = _tensorize_for_training(in_)
                for key in ["input_ids", "attention_mask", "token_type_ids"]:
                    inputs[key] += out[key].numpy().tolist()
        else:
            with Pool(self.n_process) as p:
                for out in p.imap_unordered(_tensorize_for_training, sharded_inputs):
                    for key in ["input_ids", "attention_mask", "token_type_ids"]:
                        inputs[key] += out[key].numpy().tolist()

        N = len(inputs["input_ids"])
        indices = np.random.permutation(range(N))
        for k, v in inputs.items():
            inputs[k] = np.array(v)[indices]
        n_per_shard = math.ceil(N / self.n_gpu)

        for i, _path in enumerate(all_tensorize_paths):
            start = i*n_per_shard
            end = (i+1)*n_per_shard
            curr_inputs = {k:v[start:end].tolist() for k, v in inputs.items()}
            with open(_path, "wb") as f:
                pkl.dump(curr_inputs, f)
            self.logger.info("Preprocessing done for i=%d" % i)

        self.logger.info("Finish saving preprocessed data ...")
    

    def load_tensorized_data(self, train_data, keyword, seed):
        assert self.tensorize_dir is not None

        if not os.path.exists(self.tensorize_dir):
            os.makedirs(self.tensorize_dir)

        method_name = self.method
        k_name = "%d" %  len(train_data)
        length_name = "%d-%d" % (self.n_prefix_tokens, self.max_length)
        postfix = ""

        tensorize_path = os.path.join(self.tensorize_dir,
                                      "{}_{}_k={}_seed={}_length={}{}-rank=%d.pkl".format(
                                          keyword, method_name, k_name, seed, length_name,
                                          postfix))

        if self.local_rank==-1:
            self.logger.info(tensorize_path)
        else:
            self.logger.info(tensorize_path % self.local_rank)
        all_tensorize_paths = [tensorize_path % i for i in range(self.n_gpu)]


        if not np.all([os.path.exists(_path) for _path in all_tensorize_paths]):
            self.logger.info("Tensorization was not done. Run with `--do_tensorize` without distributed mode"
                        "and then run training command again")
            raise NotImplementedError()

        if self.local_rank==-1:
            inputs = defaultdict(list)
            for i in range(self.n_gpu):
                with open(tensorize_path % i, "rb") as f:
                    curr_inputs = pkl.load(f)
                for k, v in curr_inputs.items():
                    inputs[k] += v
        else:
            assert 0<=self.local_rank<self.n_gpu
            with open(tensorize_path % self.local_rank, "rb") as f:
                inputs = pkl.load(f)

        self.tensorized_inputs = inputs
    
    

    
    def get_dataloader(self, batch_size):
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

        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])  
            
        train_sampler = RandomSampler(train_dataset)
        val_sampler = RandomSampler(val_dataset)
        
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)
        
        
        return train_dataloader,val_dataloader

        
def append_prefix(ids1, ids2, max_length, n_prefix_tokens=0,
    prefix_token_ids=None, prefix=True, allow_truncation=True, task=None):

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

    if n_prefix_tokens > 0:
        if task is None:
            _prefix_token_ids = prefix_token_ids
        else:
            _prefix_token_ids = prefix_token_ids[task]

        if prefix:
            ids1 = _prefix_token_ids + ids1
        else:
            ids1 += ids2
            ids2 = _prefix_token_ids

    n_mask = max_length-len(ids1)-len(ids2)
    assert n_mask>=0, (max_length, len(ids1), len(ids2))
    input_ids = ids1+ids2+[0 for _ in range(n_mask)]
    attention_mask = [1 for _ in ids1+ids2] + [0 for _ in range(n_mask)]
    token_type_ids = [0 for _ in ids1] + [1 for _ in ids2] + [0 for _ in range(n_mask)]
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
            data_path = "data/mbbp/mbbp_train.jsonl"
        # obtain name of data path where dataset jsonl file stored


        # if split=='train':
        #     if full_train:
        #         data_path = os.path.join(data_dir, dataset, "{}_full_train.jsonl".format(dataset))
        #     else:
        #         data_path = os.path.join(data_dir, dataset,
        #                 "{}_{}_{}_{}.jsonl".format(dataset, k, seed, split))
        # elif split=='dev':
        #     data_path = os.path.join(data_dir, dataset,
        #                 "{}_{}_{}_{}.jsonl".format(dataset, k, seed, split))
        # elif split=='test':
        #     data_path = os.path.join(data_dir, dataset,
        #                 "{}_test.jsonl".format(dataset))
        # else:
        #     print("choose split from [train, dev, test]")
        #     exit(1)


        ###load jsonl file and return a list of dictionaries(each dict represents one entry in dataset)
        with open(data_path, "r") as f:
            for line in f:
                dp = json.loads(line)
                
                if is_null:
                    dp["input"] = "N/A"
                data.append(dp)

    return data