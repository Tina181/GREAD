# cd src
# export CUDA_VISIBLE_DEVICES=3
# can use args: --use_best_params

for dataset in Pubmed Cora Citeseer; do
    echo $dataset
    python run_GNN.py --dataset $dataset 
done