
for MODEL_SEED in 0 1 2 3 4
do 
    CUDA_VISIBLE_DEVICES=7 python train.py --dataset1 cifar10 --lr1 0.1 --sched triangle --seed 0 --model_seed $MODEL_SEED --model_type resnet9 --num_epochs 30 --name standard
done

for MODEL_SEED in 0 1 2 3 4
do 
    CUDA_VISIBLE_DEVICES=7 python train.py --dataset1 cifar10 --lr1 0.1 --sched triangle --seed 0 --model_seed $MODEL_SEED --model_type resnet9 --num_epochs 30 --name reverse --reverse_splits 1
done