#!/bin/bash
#python train_model.py --type 'point' --load_feat --min_loss_delta 0.001 --max_patience 30 --save_freq 100 --lr 1e-5 --tag 1500  --n_points 1500 --max_iter 200
python eval_model.py --load_feat  --load_checkpoint --type 'point' --tag 1500 --n_points 1500
#python train_model.py --type 'point' --load_feat --min_loss_delta 0.001 --max_patience 30 --save_freq 100 --lr 1e-5 --tag 1750  --n_points 1750 --max_iter 200
python eval_model.py --load_feat  --load_checkpoint --type 'point' --tag 1750 --n_points 1750

