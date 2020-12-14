DRL code based on https://github.com/openai/universe-starter-agent

It is for the training code of the problem of active tracking discussed in the [following paper](https://sites.google.com/site/whluoimperial/active_tracking_icml2018).

@inproceedings{luo2018end,  
title={End-to-end Active Object Tracking via Reinforcement Learning},  
author={Luo, Wenhan and Sun, Peng and Zhong, Fangwei and Liu, Wei and Zhang, Tong and Wang, Yizhou},  
booktitle={International Conference on Machine Learning},  
year={2018}  
}

# Dependencies
See https://github.com/openai/universe-starter-agent

The followings are optional

* [gym_tvizdoom](https://bitbucket.org/pengsun000/gym-tvizdoom/src/master/) (for vizdoom related envs)
* [gym_unrealcv](https://github.com/zfw1226/gym-unrealcv) (for object tracking based on unreal engine envs)

# How to run

##### vizdoom example
python train.py --num-workers=2 --env-id=TrackObjSmallMazeRandFlip-v2 --log-dir=save/to --max-global-steps=29999 --val-model-secs=30 --max-val-episodes=2

python evaluate.py --env-id=TrackObjCounterclockwise-v2 --ckpt-dir=save_gallery/tosmrandflip-t8-v2-val --sleep-time=0.00 --max-episodes=10 --render --verbose=1


##### UnrealCV example
python train.py --num-workers=1 --env-id=Active-Tracking-Discrete-v0 --env-val-id=Active-Tracking-Discrete-v0 --model-id=convx2lstm --lr=0.00005 --log-dir=save/tmp --max-global-steps=250000000 --val-model-secs=-1 --max-val-episodes=40

python train.py --num-workers=1 --env-id=Tracking-Indoor1JasperPath1Static-v0 --env-val-id=Tracking-Indoor1JasperPath1Static-v0 --model-id=convx2lstm_small --lr=0.00005 --max-global-steps=250000000 --log-dir=save/tmp --val-model-secs=-1 --max-val-episodes=40


