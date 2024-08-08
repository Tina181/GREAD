# cd src
# export CUDA_VISIBLE_DEVICES=3
# cornell texas wisconsin can use args: --use_best_params

for dataset in cornell texas wisconsin ogbn-arxiv; do
    echo $dataset
    python run_GNN.py --dataset $dataset 
done