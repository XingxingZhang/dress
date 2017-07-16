
# I used simplicity of two levels
# note you must use the model EncDecAWE
ID=2

. ~/conf/cudnn.sh

codedir=/afs/inf.ed.ac.uk/group/project/img2txt/rewriting_model/dress/dress

curdir=`pwd`
lr=0.001
label=.256.2L.autoencoder
model=$curdir/model_$lr$label.t7
log=$curdir/log_$lr$label.txt
validout=$curdir/valid.out$label
testout=$curdir/test.out$label

train=/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/wiki-full/norm_all/auto-encoder-data/train.txt
valid=/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/wiki-full/norm_all/auto-encoder-data/valid.txt
test=/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/wiki-full/norm_all/auto-encoder-data/test.txt

src_vocab=/disk/scratch1/xingxing.zhang/seq2seq/sent_simple/encdec_rl-wiki-full/auto_encoder/res/wiki.full.aner.train.src.vocab.tmp.t7
dst_vocab=/disk/scratch1/xingxing.zhang/seq2seq/sent_simple/encdec_rl-wiki-full/auto_encoder/res/wiki.full.aner.train.dst.vocab.tmp.t7


cd $codedir

CUDA_VISIBLE_DEVICES=$ID th train_auto_encoder.lua --useGPU \
    --model EncDec \
    --attention dot \
    --ori_src_vocab_path $src_vocab \
    --ori_dst_vocab_path $dst_vocab \
    --seqLen 85 \
    --freqCut 3 \
    --nhid 256 \
    --nin 256 \
    --nlayers 2 \
    --dropout 0.2 \
    --lr $lr \
    --valid $valid \
    --test $test \
    --optimMethod Adam \
    --save $model \
    --train $train \
    --validout $validout --testout $testout \
    --batchSize 32 \
    --validBatchSize 32 \
    --maxEpoch 30 \
    --embedOption fineTune \
    --fineTuneFactor 0 \
    | tee $log

cd $curdir


