source_path=$1
source_mdl=$2
dir=$3

mkdir -p $dir/make_mdl
cp $source_path/$source_mdl $dir/make_mdl
cp $source_path/final.mdl $dir/make_mdl
#cp local/chain/scripts/prepare_btdnn_mdl.py $dir/make_mdl

nnet3-am-copy --binary=false $dir/make_mdl/$source_mdl $dir/make_mdl/mid.mdl
nnet3-am-copy --binary=false $dir/make_mdl/final.mdl $dir/make_mdl/final_txt.mdl
#nnet3-am-copy --binary=false 0.mdl make_mdl/0.mdl

#cd make_mdl
python local/chain_kaldi_feats/scripts/prepare_btdnn_7q_raw.py $dir
#python local/chain/prepare_btdnn_7q_raw.py

#cd ..
mv $dir/0.raw $dir/0_ori.raw

cp $dir/make_mdl/prior.raw $dir/0.raw
