'''
Load LLM on which we want to infer for given dataset
'''



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

class MetaICLModel(object):

    def __init__(self, gpt2="gpt2-large", logger=None, 
         fp16=False, local_rank=-1, soft_prefix=False, n_tokens=10, 
        prefix_embed_file=None, task_counts=None):
        if logger is None:
            class Logger():
                def info(self, text):
                    print ("Logging from MetaICLModel:\t", text)
            logger = Logger()

        self.logger = logger
        
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

      
        model = AutoModelForCausalLM.from_pretrained(gpt2,trust_remote_code=True)
        self.model_name = gpt2

        if torch.__version__ == '1.14.0.dev20221208+cu117':
            self.model = torch.compile(model)
        else:
            self.model = model 

 
    
    def perform_inference(self, data, batch_size=1):
            #batch_size =1 
            
            num_return_seq =5
            dataloader = data.get_dataloader(batch_size, is_training=False)
            
            losses = []
            preds = []

            

            data.tokenizer.padding_side = "left" 
            if data.tokenizer.pad_token == None:
                data.tokenizer.pad_token = data.tokenizer.eos_token

            iter =0
            for batch in tqdm(dataloader):
                input_ids=batch[0].to(self.device)
                attention_mask=batch[1].to(self.device)
                token_type_ids=batch[2].to(self.device)

                
                
                
                with torch.no_grad():
                    outputs = self.model.generate(input_ids=input_ids, pad_token_id=data.tokenizer.eos_token_id, attention_mask=attention_mask,max_new_tokens = 200,do_sample=True,temperature = 0.7,num_return_sequences = num_return_seq,top_k = 50)
                    
                    # print(outputs.shape) #[10,2000]
                    # print(input_ids.shape)  #[2,1800]
                    #print("\n",len(outputs))
                    #print(type(outputs))
                    decoded_outputs = data.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                    
                    for idx in range(batch_size):
                        decoded_inputs = data.tokenizer.decode(input_ids[idx], skip_special_tokens=True)
                        dp= {}
                        dp["input"] = decoded_inputs
                        dp["output"]  = decoded_outputs[idx*num_return_seq:(idx+1)*num_return_seq]
                        preds.append(dp)
                        # for i in range(len(outputs)):
                            
                            
                        #     dp ={}
                        #     dp["input"] = decoded_inputs
                        #     dp["output"] = decoded_outputs

                iter+=1

                # if iter == 5:
                #   break      
                    
                    

            return preds ###list of dicts , each having i/p & o/p



            
    

    def do_predict(self, data, batch_size=1, losses=None, verbose=False, return_nll=False):
        if losses is None:
            losses = self.do_inference(data, batch_size, verbose=verbose)
        losses = np.array(losses)
        assert len(losses)==len(data)
        predictions = []
        all_nlls = []
        gt_labels = []
        pred_labels = []

        ###sort the predictions based on losses
        for idx, dp in enumerate(data.metadata):
            curr_label_losses = [np.sum(losses[indices]) for indices in dp["indices"]]
            all_nlls.append(curr_label_losses)
            gt_labels.append(dp["label"])
            prediction_idx = sorted(enumerate(curr_label_losses), key=lambda x: x[1])[0][0]
            pred_labels.append(prediction_idx)
            prediction = dp["options"][prediction_idx]
            predictions.append(prediction.strip())
        if return_nll:
            return predictions, all_nlls, np.array(gt_labels), np.array(pred_labels)
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

def setup_fp16(model, optimizer):
    try:
        import apex
        from apex import amp
        apex.amp.register_half_function(torch, "einsum")
    except ImportError:
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    fp16_opt_level = "O1"
    model, optimizer = amp.initialize(model, optimizer, opt_level=fp16_opt_level)
    return model, optimizer



