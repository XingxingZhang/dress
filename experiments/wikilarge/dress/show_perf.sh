


codedir=/afs/inf.ed.ac.uk/group/project/img2txt/rewriting_model/encdec-rf-1.3
curdir=`pwd`

cd $codedir

# dir=/disk/scratch1/xingxing.zhang/seq2seq/sent_simple/encdec_rl/test_lm_sim_symtric_sari/model_0.01.sym-sari1.lm0.5.sim0.25.rev0.9.def.lr0.01.rf2
# dir=/disk/scratch1/xingxing.zhang/seq2seq/sent_simple/encdec_rl-wiki/simple_wiki_only/model_0.01.sym-sari1.lm0.5.sim0.25.rev0.9.lr0.01.rf2
dir=/disk/scratch1/xingxing.zhang/seq2seq/sent_simple/encdec_rl-wiki-full/rl/model_0.01.sym-sari1.lm0.5.sim0.25.rev0.9.lr0.01.rf2
log=$curdir/`basename $dir`.log.txt
log_greedy=$curdir/`basename $dir`.greedy.log.txt
echo $log
echo $log_greedy


CUDA_VISIBLE_DEVICES=3 th show_rf_valid_perf.lua --dir $dir | tee $log
CUDA_VISIBLE_DEVICES=3 th show_rf_valid_perf.lua --dir $dir --greedy | tee $log_greedy


cd $curdir



