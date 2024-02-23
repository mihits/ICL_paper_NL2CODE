# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM

class DemoInferModel(object):

    def __init__(self, gpt2="gpt2-large", logger=None, 
        out_dir=None, fp16=False, local_rank=-1, soft_prefix=False, n_tokens=10, 
        prefix_embed_file=None, task_counts=None):
        if logger is None:
            class Logger():
                def info(self, text):
                    print ("Logging from MetaICLModel:\t", text)
            logger = Logger()

        self.logger = logger
        self.out_dir = out_dir
        self.fp16 = fp16
        self.local_rank = local_rank

        if self.local_rank == -1:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            n_gpu = torch.cuda.device_count()
            ws = 1
        else:  # distributed mode
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            ws = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))
            torch.distributed.init_process_group(backend="nccl")
            n_gpu = 1

        self.n_gpu = n_gpu
        self.device = device
        if self.local_rank <= 0:
            logger.info("Setting up for local_rank=%d, world_size=%d" % (self.local_rank, ws))
        self.model_name = None
        self.model = None
        self.mode = None
        self.load(gpt2)
        self.soft_prefix = soft_prefix
        if soft_prefix:
            if task_counts is None:
                self.n_tokens = n_tokens
            else:
                self.n_tokens = n_tokens * len(task_counts)
            self.orig_vocab_size = self.model.get_input_embeddings().weight.size(0)
            print("original vocab size: ", self.orig_vocab_size)
            self.model.resize_token_embeddings(self.orig_vocab_size + self.n_tokens)
            self.new_vocab_size = self.model.get_input_embeddings().weight.size(0)
            assert self.new_vocab_size == self.n_tokens + self.orig_vocab_size
            if prefix_embed_file is not None:
                self.model.set_input_embeddings(torch.load(prefix_embed_file, map_location= self.device))
            else:
                self.model.get_input_embeddings().weight.data[-self.n_tokens:] = \
                    self.model.get_input_embeddings().weight.data[:self.n_tokens]
            self.model.tie_weights()
                    

    def __str__(self):
        text = "[MetaICL Model]: "
        if self.model_name is None:
            text += "No model loaded yet"
        else:
            text += self.model_name
            if self.mode is None:
                text += " (no mode setted - try .train() or .eval()"
            else:
                text += " (%s mode)" % self.mode
        text += "\nusing device %s, %d gpus, local_rank=%d" % (self.device, self.n_gpu, self.local_rank)
        return ("="*50) + "\n" + text + "\n" + ("="*50)

    def is_none(self):
        return self.model is None

    def eval(self):
        self.model.eval()
        self.mode = "eval"

    def cuda(self):
        self.model.cuda()

    def to_device(self):
        self.model.to(self.device)

    def load(self, gpt2="gpt2-large"):
        model = AutoModelForCausalLM.from_pretrained(gpt2)
        self.model_name = gpt2

        if torch.__version__ == '1.14.0.dev20221208+cu117':
            self.model = torch.compile(model)
        else:
            self.model = model 


    def do_inference(self, data, batch_size=1, verbose=True):
        dataloader = data.get_dataloader(batch_size, is_training=False)
        if verbose:
            dataloader = tqdm(dataloader)
        losses = []
        for batch in dataloader:
            # # input_ids=batch[0].cuda()
            # # attention_mask=batch[1].cuda()
            # # token_type_ids=batch[2].cuda()

            

            input_ids=batch[0].to(self.device)
            attention_mask=batch[1].to(self.device)
            token_type_ids=batch[2].to(self.device)
            if len(batch)==3:
                labels=None
            else:
                labels=batch[3].cuda()
            with torch.no_grad():
                loss = self.run_model(input_ids, attention_mask, token_type_ids, labels=labels)
            losses += loss.cpu().detach().numpy().tolist() 
        return losses

    def do_predict(self, data, batch_size=1, losses=None, verbose=False, return_nll=False):
        if losses is None:
            losses = self.do_inference(data, batch_size, verbose=verbose)
        losses = np.array(losses)
        assert len(losses)==len(data)
        predictions = []
        all_nlls = []


        ###sort the predictions based on losses
        for idx, dp in enumerate(data.metadata):
            curr_label_losses = [np.sum(losses[indices]) for indices in dp["indices"]]
            all_nlls.append(curr_label_losses)
            
        if return_nll:
            return  all_nlls
        else:
            return predictions

    def run_model(self, input_ids, attention_mask, token_type_ids, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[..., :-1, :].contiguous()

        if labels is None:
            labels = input_ids
        labels = labels[..., 1:].contiguous()
        label_mask = token_type_ids[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        losses = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1)) # [batch_size, length]

        losses = losses.view(logits.size(0), logits.size(1)) * label_mask
        return torch.sum(losses, axis=1) / torch.sum(label_mask, axis=1)




