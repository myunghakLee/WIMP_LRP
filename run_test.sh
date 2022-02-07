python test.py --mode val --dataroot data/argoverse_processed_simple --IFC \
--lr 0.0001 --weight-decay 0.0 --non-linearity relu  --use-centerline-features \
--segment-CL-Encoder-Prob --num-mixtures 6 --output-conv --output-prediction \
--gradient-clipping --hidden-key-generator --k-value-threshold 10 \
--scheduler-step-size 60 90 120 150 180  --distributed-backend ddp \
--experiment-name example_test --gpus 3 --batch-size 200 \
--ckpt_path /workspace/MotionPrediction/WIMP/KU_result/epoch=113.ckpt \
--save_dir ResultsJson/WIMP_KU --workers 12 --is_valid
