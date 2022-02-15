python main.py --mode train --dataroot ./data/argoverse_processed_simple --IFC \
--lr 0.0001 --weight-decay 0.0 --non-linearity relu  --use-centerline-features \
--segment-CL-Encoder-Prob --num-mixtures 6 --output-conv --output-prediction \
--gradient-clipping --hidden-key-generator --k-value-threshold 10 \
--scheduler-step-size 60 90 120 150 180  --distributed-backend ddp \
--experiment-name example --gpus 1 --batch-size 25
