dataset=Cora
reaction_term=a
echo $dataset
echo $reaction_term
python run_GNN.py --dataset $dataset --reaction_term $reaction_term 