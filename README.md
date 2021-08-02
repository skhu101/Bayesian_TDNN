# Bayesian_TDNN
This repository contains the Kaldi LF-MMI implementation of the paper **Bayesian Learning of 
LF-MMI Trained Time Delay Neural Networks for Speech Recognition**, TASLP 2021.

By Shoukang Hu, Xurong Xie, Shansong Liu, Jianwei Yu, Zi Ye, Mengzhe Geng, Xunying Liu, Helen Meng

## Getting Started
* Install [Kaldi](https://github.com/kaldi-asr/kaldi)
* Clone the repo:
  ```
  git clone https://github.com/skhu101/Bayesian_TDNN.git
  ```
  
### Usage
Step 1: Add the BayesTdnnV2Component to kaldi/src/nnet3/nnet-convolutional-component.h and kaldi/src/nnet3/nnet-tdnn-component.cc
Add the following four lines to the corresponding location in kaldi/src/nnet3/nnet-component-itf.cc
else if (cpi_type == "BayesTdnnV2ComponentPrecomputedIndexes") {
    ans = new BayesTdnnV2Component::PrecomputedIndexes();

else if (component_type == "BayesTdnnV2Component") {
    ans = new BayesTdnnV2Component();

Step 2: This part of code should be run based on the standard TDNN model (run_tdnn_7q.sh)
```shell
bash local/chain_kaldi_feats/run_btdnn_7q.sh exp/chain_kaldi_feats/btdnn7q_sp_4epoch (directory of the standard TDNN system) 1200.mdl (TDNN model updated with half of the total iterations)
```

Result comparison:
tdnn_7q
9.6 (swbd in hub5' 00), 18.0(callhm in hub5' 00), 13.8 (total in hub5' 00)
12.3(fisher in rt03), 20.0(swbd in rt03), 16.3 (total in rt03), 

bayes_tdnn_7q
9.4 (swbd in hub5' 00), 17.3(callhm in hub5' 00), 13.4 (total in hub5' 00)
11.7(fisher in rt03), 19.3(swbd in rt03), 15.7 (total in rt03)
