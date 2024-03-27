#--use demonstrations must always be present
TRAIN_METHOD=direct
TEST_METHOD=direct
LR=1e-2
N_PREFIX=10
DATASET=humaneval,mbbp
TRAIN_TASK=code_dataset
SPLIT=train
MODEL=bigcode/santacoder
TRAIN_SIZE=100
STEP=1000
K=4

CUDA_VISIBLE_DEVICES=0 python task_inference.py\
    --dataset $DATASET\
    --gpt $MODEL\
    --method $TEST_METHOD\
    --test_batch_size 2\
    --k $K\
    --embedding_dir embeddings/\
    --use_demonstrations\
    --concept_temperature 50\
    --similarity_temperature 0.1\
    --train_size $TRAIN_SIZE\
    --difficulty concept_calibrated\
    --n_prefix_tokens $N_PREFIX\
    --concept_dir concept_likelihood/gpt2/$TRAIN_TASK-$SPLIT-$TRAIN_SIZE/$DATASET-$TRAIN_METHOD-prefix=$N_PREFIX-lr=$LR-$STEP\
    --prefix_embed_file checkpoints/gpt2/$TRAIN_TASK-$SPLIT/prefix={$N_PREFIX}-{$TRAIN_METHOD}-lr={$LR}-initByVocab/soft_embeddings-$STEP.pt\
    --prior easiest\
    --seed 100\
    --max_length 1800\  ##size of model context window - max_new_tokens generated
    --use_similar_demo\
     #--use_random_demo\
       # --reorder\
    # --prior most_similar\

# CUDA_VISIBLE_DEVICES=0 python task_inference.py\
#     --dataset $DATASET\      
#     --method $TEST_METHOD\
#     --gpt $MODEL\
#     --test_batch_size 4\
#     --k $K\
#     --embedding_dir embeddings/\
#     --use_demonstrations\
#     --concept_temperature 50\
#     --similarity_temperature 0.1\
#     --train_size $TRAIN_SIZE\
#     --difficulty concept_calibrated\
#     --n_prefix_tokens $N_PREFIX\
#     --concept_dir concept_likelihood/gpt2/$TRAIN_TASK-$SPLIT-$TRAIN_SIZE/$DATASET-$TRAIN_METHOD-prefix=$N_PREFIX-lr=$LR-$STEP\
#     --prefix_embed_file checkpoints/gpt2/$TRAIN_TASK-$SPLIT/prefix={$N_PREFIX}-{$TRAIN_METHOD}-lr={$LR}-initByVocab/soft_embeddings-$STEP.pt\
#     --prior easiest\
#     --seed 100\
#     --max_length 1024\  ##size of model context window
#    # --reorder\
#     # --prior most_similar\