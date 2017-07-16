
codedir=/afs/inf.ed.ac.uk/group/project/img2txt/rewriting_model/dress/dress
curdir=`pwd`
GPUID=0

cd $codedir


model=/disk/scratch1/xingxing.zhang/seq2seq/sent_simple/encdec_rl-wiki-full/baseline/model_0.001.256.2L.we.full.2l.ft0.t7
train=/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/wiki-full/norm_all/wiki.full.aner.train
valid=/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/wiki-full/norm_all/wiki.full.aner.valid
test=/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/wiki-full/norm_all/wiki.full.aner.test



output=$curdir/wiki.full.align.t7
log=$curdir/aligner.log.txt


CUDA_VISIBLE_DEVICES=$GPUID th aligner.lua \
    --modelPath $model \
    --train $train \
    --valid $valid \
    --test $test \
    --output $output \
    --showAlign \
    | tee $log

cd $curdir



