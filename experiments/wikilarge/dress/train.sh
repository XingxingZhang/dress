
ID=`./gpu_lock.py --id-to-hog 3`
# I used simplicity of two levels
# note you must use the model EncDecAWE
# ID=3
echo $ID
if [ $ID -eq -1 ]; then
    echo "this gpu is not free"
    exit
fi

. ~/conf/cudnn.sh

codedir=/afs/inf.ed.ac.uk/group/project/img2txt/rewriting_model/dress/dress
curdir=`pwd` 
lr=0.01
label=.sym-sari1.lm0.5.sim0.25.rev0.9.lr0.01.rf2
model_dir=$curdir/model_$lr$label
log=$curdir/log_$lr$label.txt
validout=$curdir/valid.out$label
testout=$curdir/test.out$label

train=/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/wiki-full/norm_all/wiki.full.aner.train
valid=/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/wiki-full/norm_all/wiki.full.aner.valid
test=/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/wiki-full/norm_all/wiki.full.aner.test

wembed=/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/wiki-full/norm_all/glove.840B.300d.wiki.full.aner.t7

# pre-trained encoder-decoder model path
submodel=/disk/scratch1/xingxing.zhang/seq2seq/sent_simple/encdec_rl-wiki-full/baseline/model_0.001.256.2L.we.full.2l.ft0.t7

# pre-trained LSTM language model path
lmpath=/disk/scratch1/xingxing.zhang/seq2seq/sent_simple/encdec_rl-wiki-full/lm/model_1.0.sgd.t7
# pre-trained auto_encoder path
simpath=/disk/scratch1/xingxing.zhang/seq2seq/sent_simple/encdec_rl-wiki-full/auto_encoder/model_0.001.256.2L.t7

cd $codedir

CUDA_VISIBLE_DEVICES=$ID th train_rf.lua --learnZ --useGPU \
    --model EncDecARF_Plus \
    --attention dot \
    --seqLen 85 \
    --freqCut 3 \
    --nhid 256 \
    --nin 300 \
    --nlayers 2 \
    --dropout 0.2 \
    --lr $lr \
    --optimMethod SGD \
    --valid $valid \
    --test $test \
    --save $model_dir \
    --train $train \
    --validout $validout --testout $testout \
    --batchSize 16 \
    --validBatchSize 16 \
    --maxEpoch 100 \
    --wordEmbedding $wembed \
    --embedOption fineTune \
    --fineTuneFactor 0 \
    --encdecPath $submodel \
    --sampleStart 25 \
    --deltaSamplePos 3 \
    --rfEpoch 2 \
    --lmPath $lmpath \
    --simPath $simpath \
    --sariWeight 1 \
    --lmWeight 0.5 \
    --simWeight 0.25 \
    --sariRevWeight 0.9 \
    | tee $log

cd $curdir

./gpu_lock.py --free $ID
./gpu_lock.py


