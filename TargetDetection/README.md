## how to train

nohup python -u train.py --resume weights/ssd300_Xray20190723_10100.pth --start_iter 10100 &


## how to test
#now should change eval_5_epoch_for.py EPOCHS
# nohup python eval_5epoch_for.py --trained_model 10100  > result10100.text &
nohup python eval_5epoch_for.py --SIXray_root data/core_500/  > result.text &
