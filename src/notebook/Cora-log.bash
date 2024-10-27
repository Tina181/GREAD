dataset="Cora"
reaction_term="aggdiff-log"
echo "dataset:" $dataset
echo "reaction_term:" $reaction_term

python run_GNN.py --dataset $dataset --gpu=4 \
                --reaction_term $reaction_term --time=1 --step_size=1\
                --max_nfe=5000 \
                --log_eps=0.2\
                --method='euler'
