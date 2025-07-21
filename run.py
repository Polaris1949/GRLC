import os

if __name__ == "__main__":
    # os.system(r"python train.py --dataset-class Planetoid --dataset-name Cora --custom-key classification")
    os.system(r"python train_av2.py --epochs 1000000 --data-root . --dataset-class Yamai --dataset-name Argoverse2 --custom-key classification")
