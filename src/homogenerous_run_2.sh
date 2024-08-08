# cd src
# export CUDA_VISIBLE_DEVICES=3
# can not add args: --use_best_params
for dataset in Computers Photo CoauthorCS; do
    echo $dataset
    python run_GNN.py --dataset $dataset 
done