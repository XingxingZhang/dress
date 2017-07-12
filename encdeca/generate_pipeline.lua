
require '.'
require 'shortcut'
require 'sampleA'
require 'post_processing_unk'

local function main()
  local cmd = torch.CmdLine()
  cmd:text('Options for generate raw output:')
  cmd:option('--modelPath',
    '/disk/scratch/XingxingZhang/encdec/sent_simple/encdec_attention_PWKP_margin_prob/model_0.001.256.dot.2L.adam.reload.sgd.m0.97.t7',
    'model path')
  cmd:option('--dataPath',
    '/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/PWKP/an_ner/PWKP_108016.tag.80.aner.valid',
    'data path. Note that it should be the path before .src and .dst')
  cmd:option('--outPathRaw',
'/disk/scratch/XingxingZhang/encdec/sent_simple/encdec_attention_PWKP_margin_prob/sampleA/model_0.001.256.dot.2L.adam.reload.sgd.m0.97.valid', 
    'raw output path. Note the current output is without UNK replacement and NER recovery')
  cmd:option('--oriDataPath',
    '/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/PWKP/an_ner/PWKP_108016.tag.80.aner.ori.valid', 
    'original data path without ner replacement')
  cmd:option('--oriMapPath',
    '/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/PWKP/an_ner/PWKP_108016.tag.80.aner.map.t7',
    'map between NER and original text')
  
  local opts = cmd:parse(arg)
  
  -- generate raw output by sampling (the output without UNK replacement and NER recovery)
  local sampler = EncDecASampler(opts.modelPath)
  sampler:generateBatch(opts.dataPath, opts.outPathRaw)
  
  local att_file = opts.outPathRaw .. '.att.t7'
  local unk_rep_file = opts.outPathRaw .. '.unk.rep.txt'
  local ner_src_file = opts.dataPath .. '.src'
  PostProcessorUnk.replaceUnk(ner_src_file, opts.outPathRaw, att_file, unk_rep_file)
  
  local out_file = opts.outPathRaw .. '.out.txt'
  local ref_file = opts.oriDataPath .. '.dst'
  PostProcessorUnk.recoverNER(ner_src_file, unk_rep_file, opts.oriMapPath, out_file)
  
  -- local cmd = string.format('./scripts/multi-bleu.perl %s < %s', ref_file, out_file)
  -- os.execute(cmd)
  
  local src_file = opts.oriDataPath .. '.src'
  local bleu_eval = require 'bleu_eval'
  local bleu = bleu_eval.eval(src_file, ref_file, out_file)
  printf('bleu = %f\n', bleu)
  
  local fkgl_eval = require 'fkgl_eval'
  local fkgl = fkgl_eval.eval(out_file)
  printf('FKGL = %f\n', fkgl)
  
  PostProcessorUnk.compareOri(src_file, out_file)
  
  local analyze_results = require 'analyze_results'
  local ana_file = opts.outPathRaw .. '.ana.txt'
  analyze_results.analyze(src_file, ref_file, out_file, ana_file)
  printf('analyze file save at %s\n', ana_file)
end

main()


