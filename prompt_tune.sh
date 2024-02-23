METHOD=direct
LR=1e-2
N_PREFIX=10
TASK=code_dataset
SPLIT=train
MODEL=gpt2
STEP=5000
CUDA_VISIBLE_DEVICES=0 python prompt_tune.py\
    --gpt2 $MODEL\
    --task $TASK\
    --split $SPLIT\
    --method $METHOD\
    --save_period 1000\
    --log_period 2\
    --seed 100\
    --n_gpu 1\
    --tensorize_dir tensorized\
    --out_dir checkpoints/$MODEL/$TASK-$SPLIT/prefix={$N_PREFIX}-{$METHOD}-lr={$LR}-initByVocab\
    --batch_size 4\
    --lr $LR\
    --n_prefix_tokens $N_PREFIX\
    --num_training_steps 5\
    --n_process 10\
    --do_tensorize
#  --prefix_embed_file checkpoints/gpt2/$TRAIN_TASK-$SPLIT/prefix={$N_PREFIX}-{$TRAIN_METHOD}-lr={$LR}-initByVocab/soft_embeddings-$STEP.pt\
    