
codedir=/afs/inf.ed.ac.uk/group/project/img2txt/rewriting_model/dress-release/encdeca
curdir=`pwd`

# your trained model
model=/disk/scratch1/xingxing.zhang/seq2seq/sent_simple/encdec_rl-wiki-full/baseline/model_0.001.256.2L.we.full.2l.ft0.t7
data=/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/wiki-full/norm_all/wiki.full.aner
output=$curdir/out.t7
oridata=/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/wiki-full/norm_all/wiki.full.aner.ori
orimap=/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/wiki-full/norm_all/wiki.full.aner.map.t7
log=$output.log

cd $codedir

CUDA_VISIBLE_DEVICES=3 th generate_pipeline.lua \
    --modelPath $model \
    --dataPath $data.valid \
    --outPathRaw $output.valid \
    --oriDataPath $oridata.valid \
    --oriMapPath $orimap | tee $log.valid

CUDA_VISIBLE_DEVICES=3 th generate_pipeline.lua \
    --modelPath $model \
    --dataPath $data.test \
    --outPathRaw $output.test \
    --oriDataPath $oridata.test \
    --oriMapPath $orimap | tee $log.test

cd $curdir

