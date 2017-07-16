
# decoder starts at the second word, first one is not predictable
# WITHOUT word embedding init
# with stochastic alignment

codedir=/afs/inf.ed.ac.uk/group/project/img2txt/rewriting_model/dress/dress
curdir=`pwd`
ID=0

lr=0.001
label=.ls.soft.nowe
model=$curdir/model_$lr$label.t7
log=$curdir/log_$lr$label.txt

# src_vocab=/disk/scratch1/xingxing.zhang/seq2seq/sent_simple/encdec_lm_neu_tm/V0V4_V1V4_V2V4_V3V4_V0V3_V0V2_V1V3.aner.train.src.vocab.tmp.t7
# src_vocab=/disk/scratch1/xingxing.zhang/seq2seq/sent_simple/encdec_rl-wiki/neu_trans_model/res/PWKP_108016.tag.80.aner.train.src.vocab.tmp.t7
src_vocab=/disk/scratch1/xingxing.zhang/seq2seq/sent_simple/encdec_rl-wiki-full/neu_trans_model/res/wiki.full.aner.train.src.vocab.tmp.t7
# dst_vocab=/disk/scratch1/xingxing.zhang/seq2seq/sent_simple/encdec_lm_neu_tm/V0V4_V1V4_V2V4_V3V4_V0V3_V0V2_V1V3.aner.train.dst.vocab.tmp.t7
# dst_vocab=/disk/scratch1/xingxing.zhang/seq2seq/sent_simple/encdec_rl-wiki/neu_trans_model/res/PWKP_108016.tag.80.aner.train.dst.vocab.tmp.t7
dst_vocab=/disk/scratch1/xingxing.zhang/seq2seq/sent_simple/encdec_rl-wiki-full/neu_trans_model/res/wiki.full.aner.train.dst.vocab.tmp.t7
# dataset=/disk/scratch1/xingxing.zhang/seq2seq/sent_simple/encdec_lm_neu_tm/align/newsela.align.t7
# dataset=/disk/scratch1/xingxing.zhang/seq2seq/sent_simple/encdec_rl-wiki/neu_trans_model/align/PWKP.align.t7
dataset=/disk/scratch1/xingxing.zhang/seq2seq/sent_simple/encdec_rl-wiki-full/neu_trans_model/align/wiki.full.align.t7
# wembed=/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/newsela/all/glove.840B.300d.newsela.full.aner.t7

cd $codedir

CUDA_VISIBLE_DEVICES=$ID th train_lextrans.lua --useGPU \
    --seqLen 85 \
    --nhid 256 \
    --nin 256 \
    --nlayers 2 \
    --dropout 0.2 \
    --optimMethod Adam \
    --lr $lr \
    --srcVocab $src_vocab \
    --dstVocab $dst_vocab \
    --dataset $dataset \
    --save $model \
    --batchSize 32 \
    --maxEpoch 15 \
    --decStart 2 \
    --alignOption soft \
    | tee $log

cd $curdir





