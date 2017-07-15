
codedir=/afs/inf.ed.ac.uk/group/project/img2txt/rewriting_model/dress/dress
curdir=`pwd`
GPUID=0


model=/disk/scratch1/xingxing.zhang/seq2seq/sent_simple/encdec_rl-wiki-full/rl/model_0.01.sym-sari1.lm0.5.sim0.25.rev0.9.lr0.01.rf2/16.t7

# lm_model=/disk/scratch1/xingxing.zhang/seq2seq/sent_simple/encdec_rl-wiki/lm/model_1.0.sgd.t7
lm_model=/disk/scratch1/xingxing.zhang/seq2seq/sent_simple/encdec_rl-wiki-full/lm/model_1.0.sgd.t7
# tm_path=/disk/scratch1/xingxing.zhang/seq2seq/sent_simple/encdec_rl-wiki/neu_trans_model/model_0.001.soft.nowe.t7
tm_path=/disk/scratch1/xingxing.zhang/seq2seq/sent_simple/encdec_rl-wiki-full/neu_trans_model/model_0.001.soft.nowe.t7

# data=/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/PWKP/an_ner/PWKP_108016.tag.80.aner
data=/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/wiki-full/norm_all/wiki.full.aner
output=$curdir/out.t7
# oridata=/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/PWKP/an_ner/PWKP_108016.tag.80.aner.ori
oridata=/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/wiki-full/norm_all/wiki.full.aner.ori
# orimap=/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/PWKP/an_ner/PWKP_108016.tag.80.aner.map.t7
orimap=/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/wiki-full/norm_all/wiki.full.aner.map.t7
log=$output.log.txt


lm_weight=0
tm_weight=0.1


cd $codedir


CUDA_VISIBLE_DEVICES=$GPUID th generate_pipeline.lua \
    --modelPath $model \
    --lmPath $lm_model \
    --lmWeight $lm_weight \
    --dataPath $data.valid \
    --lexTransPath $tm_path \
    --lexTransWeight $tm_weight \
    --outPathRaw $output.valid \
    --oriDataPath $oridata.valid \
    --oriMapPath $orimap | tee $log.valid

CUDA_VISIBLE_DEVICES=$GPUID th generate_pipeline.lua \
    --modelPath $model \
    --lmPath $lm_model \
    --lmWeight $lm_weight \
    --dataPath $data.test \
    --lexTransPath $tm_path \
    --lexTransWeight $tm_weight \
    --outPathRaw $output.test \
    --oriDataPath $oridata.test \
    --oriMapPath $orimap | tee $log.test


cd $curdir



