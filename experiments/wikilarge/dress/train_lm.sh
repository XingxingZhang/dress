
ID=3

curdir=`pwd`
codedir=/afs/inf.ed.ac.uk/group/project/img2txt/rewriting_model/dress/dress
lr=1.0
label=.lm.sgd
model=$curdir/model_$lr$label.t7
log=$curdir/log_$lr$label.txt


train=/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/wiki-full/norm_all/wiki.full.aner.train.dst
valid=/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/wiki-full/norm_all/wiki.full.aner.valid.dst
test=/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/wiki-full/norm_all/wiki.full.aner.test.dst


cd $codedir
CUDA_VISIBLE_DEVICES=$ID th train_lm.lua --useGPU \
    --dropout 0.2 --batchSize 20 --validBatchSize 20 --save $model --model LSTMLM \
    --nlayers 1 \
    --lr $lr \
    --optimMethod SGD \
    --nhid 200 \
    --nin 100 \
    --minImprovement 1.001 \
    --train $train \
    --valid $valid \
    --test $test \
    | tee $log

cd $curdir


