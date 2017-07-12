
ID=`./gpu_lock.py --id-to-hog 0`
# note you must use the model EncDecAWE
# ID=3
echo $ID
if [ $ID -eq -1 ]; then
    echo "this gpu is not free"
    exit
fi
# ./gpu_lock.py

. ~/conf/cudnn.sh

codedir=/afs/inf.ed.ac.uk/group/project/img2txt/rewriting_model/dress-release/encdeca
curdir=`pwd`
lr=0.001
label=.256.2L.we.ft0
model=$curdir/model_$lr$label.t7
log=$curdir/log_$lr$label.txt
validout=$curdir/valid.out$label
testout=$curdir/test.out$label

train=/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/wiki-full/norm_all/wiki.full.aner.train
valid=/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/wiki-full/norm_all/wiki.full.aner.valid
test=/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/wiki-full/norm_all/wiki.full.aner.test


wembed=/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/wiki-full/norm_all/glove.840B.300d.wiki.full.aner.t7


cd $codedir

CUDA_VISIBLE_DEVICES=$ID th train.lua --learnZ --useGPU \
    --model EncDecAWE \
    --attention dot \
    --seqLen 85 \
    --freqCut 4 \
    --nhid 256 \
    --nin 300 \
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
    --wordEmbedding $wembed \
    --embedOption fineTune \
    --fineTuneFactor 0 \
    | tee $log


cd $curdir

./gpu_lock.py --free $ID
./gpu_lock.py

