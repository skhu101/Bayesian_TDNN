#!/bin/bash

# 7q is as 7p but a modified topology with resnet-style skip connections, more layers,
#  skinnier bottlenecks, removing the 3-way splicing and skip-layer splicing,
#  and re-tuning the learning rate and l2 regularize.  The configs are
#  standardized and substantially simplified.  There isn't any advantage in WER
#  on this setup; the advantage of this style of config is that it also works
#  well on smaller datasets, and we adopt this style here also for consistency.

# local/chain/compare_wer_general.sh --rt03 tdnn7p_sp tdnn7q_sp
# System                tdnn7p_sp tdnn7q_sp
# WER on train_dev(tg)      11.80     11.79
# WER on train_dev(fg)      10.77     10.84
# WER on eval2000(tg)        14.4      14.3
# WER on eval2000(fg)        13.0      12.9
# WER on rt03(tg)            17.5      17.6
# WER on rt03(fg)            15.3      15.2
# Final train prob         -0.057    -0.058
# Final valid prob         -0.069    -0.073
# Final train prob (xent)        -0.886    -0.894
# Final valid prob (xent)       -0.9005   -0.9106
# Num-parameters               22865188  18702628


# steps/info/chain_dir_info.pl exp/chain/tdnn7q_sp
# exp/chain/tdnn7q_sp: num-iters=394 nj=3..16 num-params=18.7M dim=40+100->6034 combine=-0.058->-0.057 (over 8) xent:train/valid[261,393,final]=(-1.20,-0.897,-0.894/-1.20,-0.919,-0.911) logprob:train/valid[261,393,final]=(-0.090,-0.059,-0.058/-0.098,-0.073,-0.073)

prior_dir=$1
prior_mdl=$2

# source /opt/share/etc/gcc-5.4.0.sh
export CUDA_VISIBLE_DEVICES=0
#cd /project_bdda4/bdda/skhu/toolkits/kaldi/egs/swbd_wav/s5c
#set -e

# configs for 'chain'
stage=0
train_stage=-10
get_egs_stage=-10
speed_perturb=true
affix=7q
if [ -e data/rt03 ]; then maybe_rt03=rt03; else maybe_rt03= ; fi

decode_iter=
decode_nj=50

# training options
frames_per_eg=150,110,100
remove_egs=false
common_egs_dir=exp/chain_kaldi_feats/tdnn7q_sp_4epoch/egs
#common_egs_dir=/project_bdda4/bdda/skhu/toolkits/kaldi/egs/swbd_wav/s5c/exp/chain/tdnn_iv_7q_sp/egs
xent_regularize=0.1
dropout_schedule='0,0@0.20,0.5@0.50,0'

test_online_decoding=false  # if true, it will run the last decoding stage.

# End configuration section.
echo "$0 $@"  # Print the command line for logging

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

suffix=
$speed_perturb && suffix=_sp
dir=exp/chain_kaldi_feats/btdnnv2${affix}${suffix}_4epo

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

# The iVector-extraction and feature-dumping parts are the same as the standard
# nnet3 setup, and you can skip them by setting "--stage 8" if you have already
# run those things.

train_set=train_nodup$suffix
ali_dir=exp/tri4_ali_nodup$suffix
treedir=exp/chain/tri5_7d_tree$suffix
lang=data/lang_chain_2y


# if we are using the speed-perturbed data we need to generate
# alignments for it.
#local/nnet3/run_ivector_common.sh --stage $stage --speed-perturb $speed_perturb --generate-alignments $speed_perturb || exit 1;


#if [ $stage -le 9 ]; then
#  # Get the alignments as lattices (gives the LF-MMI training more freedom).
#  # use the same num-jobs as the alignments
#  nj=$(cat exp/tri4_ali_nodup$suffix/num_jobs) || exit 1;
#  steps/align_fmllr_lats.sh --nj $nj --cmd "$train_cmd" data/$train_set data/lang exp/tri4 exp/tri4_lats_nodup$suffix
#  rm exp/tri4_lats_nodup$suffix/fsts.*.gz # save space
#fi
#
#
#if [ $stage -le 10 ]; then
#  # Create a version of the lang/ directory that has one state per phone in the
#  # topo file. [note, it really has two states.. the first one is only repeated
#  # once, the second one has zero or more repeats.]
#  rm -rf $lang
#  cp -r data/lang $lang
#  silphonelist=$(cat $lang/phones/silence.csl) || exit 1;
#  nonsilphonelist=$(cat $lang/phones/nonsilence.csl) || exit 1;
#  # Use our special topology... note that later on may have to tune this
#  # topology.
#  steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$lang/topo
#fi
#
#if [ $stage -le 11 ]; then
#  # Build a tree using our new topology. This is the critically different
#  # step compared with other recipes.
#  steps/nnet3/chain/build_tree.sh --frame-subsampling-factor 3 --context-opts "--context-width=2 --central-position=1" --cmd "$train_cmd" 7000 data/$train_set $lang $ali_dir $treedir
#fi

if [ $stage -le 12 ]; then
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $treedir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print(0.5/$xent_regularize)" | python)
  affine_opts="l2-regularize=0.01 dropout-proportion=0.0 dropout-per-dim=true dropout-per-dim-continuous=true"
  tdnnf_opts="l2-regularize=0.01 dropout-proportion=0.0 bypass-scale=0.66"
  linear_opts="l2-regularize=0.01 orthonormal-constraint=-1.0"
  prefinal_opts="l2-regularize=0.01"
  output_opts="l2-regularize=0.002"

  mkdir -p $dir/configs

  cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=40 name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  fixed-affine-layer name=lda input=Append(-1,0,1,ReplaceIndex(ivector, t, 0)) affine-transform-file=$dir/configs/lda.mat

  # the first splicing is moved before the lda layer, so no splicing here
  relu-batchnorm-dropout-layer name=tdnn1 $affine_opts dim=1536
  tdnnf-layer name=tdnnf2 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=1
  tdnnf-layer name=tdnnf3 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=1
  tdnnf-layer name=tdnnf4 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=1
  tdnnf-layer name=tdnnf5 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=0
  tdnnf-layer name=tdnnf6 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf7 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf8 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf9 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf10 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf11 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf12 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf13 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf14 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf15 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  linear-component name=prefinal-l dim=256 $linear_opts

  prefinal-layer name=prefinal-chain input=prefinal-l $prefinal_opts big-dim=1536 small-dim=256
  output-layer name=output include-log-softmax=false dim=$num_targets $output_opts

  prefinal-layer name=prefinal-xent input=prefinal-l $prefinal_opts big-dim=1536 small-dim=256
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor $output_opts
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/

  #cat <<EOF > $dir/configs/change.config
  #component name=tdnn1.affine type=BayesTdnnComponent input-dim=220 output-dim=1536 l2-regularize=0.01 max-change=0.75 time-offsets=0 test-flag=false use-exp-std=true share-std-output-sampling=true KL-scale=0.0001 use-natural-gradient=true prior-std=-3 bayes-dropout=false
  #component-node name=tdnn1.affine component=tdnn1.affine input=lda
#EOF

  cat <<EOF > $dir/configs/change.config
  component name=tdnn1.affine type=BayesTdnnV2Component input-dim=220 output-dim=1536 l2-regularize=0.01 max-change=0.75 time-offsets=0 test-flag=false KL-scale=0.0001 use-natural-gradient=true prior-std=-3
  component-node name=tdnn1.affine component=tdnn1.affine input=lda
EOF
  nnet3-am-copy --raw --binary=false $prior_dir/${prior_mdl} - | \
  nnet3-copy --binary=false --nnet-config=$dir/configs/change.config - $dir/0.raw
  
  bash local/chain_kaldi_feats/scripts/transfer_btdnn_7q_raw.sh $prior_dir $prior_mdl $dir

fi

if [ $stage -le 13 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{5,6,7,8}/$USER/kaldi-data/egs/swbd-$(date +'%m_%d_%H_%M')/s5c/$dir/egs/storage $dir/egs/storage
  fi

#    --cmd "queue.pl --config /home/dpovey/queue_conly.conf" \


  steps/nnet3/chain/train.py --stage $train_stage \
    --cmd "$train_cmd" \
    --feat.online-ivector-dir exp/nnet3/ivectors_${train_set} \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.0 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --trainer.dropout-schedule $dropout_schedule \
    --trainer.add-option="--optimization.memory-compression-level=2" \
    --egs.dir "$common_egs_dir" \
    --egs.stage $get_egs_stage \
    --egs.opts "--frames-overlap-per-eg 0 --constrained false" \
    --egs.chunk-width $frames_per_eg \
    --trainer.num-chunk-per-minibatch 64 \
    --trainer.frames-per-iter 1500000 \
    --trainer.num-epochs 4 \
    --trainer.optimization.num-jobs-initial 1 \
    --trainer.optimization.num-jobs-final 1 \
    --trainer.optimization.initial-effective-lrate 0.00025 \
    --trainer.optimization.final-effective-lrate 0.000025 \
    --trainer.max-param-change 2.0 \
    --cleanup.remove-egs $remove_egs \
    --trainer.input-model $dir/0.raw \
    --feat-dir data/${train_set}_hires \
    --tree-dir $treedir \
    --lat-dir exp/tri4_lats_nodup$suffix \
    --dir $dir  || exit 1;

fi

if [ $stage -le 14 ]; then
  cp $dir/final.mdl $dir/final_cp.mdl
  nnet3-am-copy --binary=false $dir/final_cp.mdl $dir/final.mdl 
  sed -i "s/<TestFlag> F/<TestFlag> T/g" $dir/final.mdl
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang_sw1_tg $dir $dir/graph_sw1_tg
fi


graph_dir=$dir/graph_sw1_tg
iter_opts=
if [ ! -z $decode_iter ]; then
  iter_opts=" --iter $decode_iter "
fi
if [ $stage -le 15 ]; then
  rm $dir/.error 2>/dev/null || true
  #for decode_set in $maybe_rt03; do
  for decode_set in eval2000 rt03; do
      (
      steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
          --nj $decode_nj --cmd "$decode_cmd" $iter_opts \
          --online-ivector-dir exp/nnet3/ivectors_${decode_set} \
          $graph_dir data/${decode_set}_hires \
          $dir/decode_${decode_set}${decode_iter:+_$decode_iter}_sw1_tg || exit 1;
      if $has_fisher; then
          steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
            data/lang_sw1_{tg,fsh_fg} data/${decode_set}_hires \
            $dir/decode_${decode_set}${decode_iter:+_$decode_iter}_sw1_{tg,fsh_fg} || exit 1;
      fi
      ) || touch $dir/.error &
  done
  wait
  if [ -f $dir/.error ]; then
    echo "$0: something went wrong in decoding"
    exit 1
  fi
fi

#if $test_online_decoding && [ $stage -le 16 ]; then
#  # note: if the features change (e.g. you add pitch features), you will have to
#  # change the options of the following command line.
#  steps/online/nnet3/prepare_online_decoding.sh \
#       --mfcc-config conf/mfcc_hires.conf \
#       $lang exp/nnet3/extractor $dir ${dir}_online
#
#  rm $dir/.error 2>/dev/null || true
#  for decode_set in train_dev eval2000 $maybe_rt03; do
#    (
#      # note: we just give it "$decode_set" as it only uses the wav.scp, the
#      # feature type does not matter.
#
#      steps/online/nnet3/decode.sh --nj $decode_nj --cmd "$decode_cmd" \
#          --acwt 1.0 --post-decode-acwt 10.0 \
#         $graph_dir data/${decode_set}_hires \
#         ${dir}_online/decode_${decode_set}${decode_iter:+_$decode_iter}_sw1_tg || exit 1;
#      if $has_fisher; then
#          steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
#            data/lang_sw1_{tg,fsh_fg} data/${decode_set}_hires \
#            ${dir}_online/decode_${decode_set}${decode_iter:+_$decode_iter}_sw1_{tg,fsh_fg} || exit 1;
#      fi
#    ) || touch $dir/.error &
#  done
#  wait
#  if [ -f $dir/.error ]; then
#    echo "$0: something went wrong in decoding"
#    exit 1
#  fi
#fi


exit 0;
