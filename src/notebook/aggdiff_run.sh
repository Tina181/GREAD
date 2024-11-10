# '''aggdiff-log aggdiff-gauss aggdiff-gat'''
dataset=Cora
method=rk4
for reaction_term in aggdiff-log aggdiff-gauss aggdiff-gat ; do
    echo $dataset
    echo $reaction_term
    echo $method
    python run_GNN.py --dataset $dataset --reaction_term $reaction_term --method $method
done
