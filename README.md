# Bayesian_TDNN
This repository contains the Kaldi LF-MMI implementation of the paper **Bayesian Learning of 
LF-MMI Trained Time Delay Neural Networks for Speech Recognition**, IEEE/ACM Transactions on Audio Speech and Language (TASLP).

By Shoukang Hu, Xurong Xie, Shansong Liu, Jianwei Yu, Zi Ye, Mengzhe Geng, Xunying Liu, Helen Meng

[Paper](https://ieeexplore.ieee.org/abstract/document/9387600)

## Getting Started
* Install [Kaldi](https://github.com/kaldi-asr/kaldi)
* Clone the repo:
  ```
  git clone https://github.com/skhu101/Bayesian_TDNN.git
  ```
  
### Usage
Step 1: 

* Add the BayesTdnnV2Component in nnet-convolutional-component.h to kaldi/src/nnet3/nnet-convolutional-component.h 

* Add the BayesTdnnV2Component in nnet-tdnn-component.cc to kaldi/src/nnet3/nnet-tdnn-component.cc 

* Add the following four lines to the corresponding location in kaldi/src/nnet3/nnet-component-itf.cc
```shell
else if (cpi_type == "BayesTdnnV2ComponentPrecomputedIndexes") {
    ans = new BayesTdnnV2Component::PrecomputedIndexes();

else if (component_type == "BayesTdnnV2Component") {
    ans = new BayesTdnnV2Component();
```

* complie the new source file 
```shell
cd kaldi/src/nnet3/
make -j 20
```

Step 2: 

run the factored TDNN model using the following command
```shell
cd kaldi/egs/swbd/s5c
bash local/chain/tuning/run_tdnn_7q.sh
```

Step 3: 

This part of code should be run based on the standard TDNN model (run_tdnn_7q.sh)
```shell
bash local/chain_kaldi_feats/run_btdnn_7q.sh \
exp/chain_kaldi_feats/btdnn7q_sp_4epoch (directory of the standard TDNN system) \
1200.mdl (TDNN model updated with half of the total iterations)
```

### Result comparison:

| Model |  hub5' 00 <br> swbd  | hub5' 00 <br> callhm  | hub5' 00 <br> avg | rt03 <br> fisher | rt03 <br> swbd | rt03 <br> avg |
| :---:   | :-: | :-: | :-: | :-: | :-: | :-: | 
| tdnn_7q | 9.6              |  18.0              | 13.8              | 12.3           | 20.0         | 16.3          |
| bayes_tdnn_7q | 9.4             |  17.3              | 13.4              | 11.7           | 19.3         | 15.7          |

Note that we set --trainer.optimization.num-jobs-initial 1 and --trainer.optimization.num-jobs-final 1 in our experiments due to computational resource constraint.

### Citation
If you find our codes or trained models useful in your research, please consider to star our repo and cite our paper:

    @article{hu2021bayesian,
      title={Bayesian Learning of LF-MMI Trained Time Delay Neural Networks for Speech Recognition},
      author={Hu, Shoukang and Xie, Xurong and Liu, Shansong and Yu, Jianwei and Ye, Zi and Geng, Mengzhe and Liu, Xunying and Meng, Helen},
      journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
      volume={29},
      pages={1514--1529},
      year={2021},
      publisher={IEEE}
    }
