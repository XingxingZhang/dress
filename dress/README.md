
== This is an implmentation of Encoder-Decoder LSTM model ==

1. add Vanilla Encoder-Decoder without Attention
  -- congratulations! this version can be used!


==v2.0==
This version I will mainly add NER support
1. add pre-processing code

==v2.5==
support UNK replacement

==v2.6==
support shuffle per epoch

==v2.6.5==
1. merge `encdec-attention-ner-unk-margin-v3.5` on
  -- view the real output
    -- 28th Feb: add bleu evaluation; using mteval-v13a.pl
    -- 28th Feb: view output
    -- 29th Feb: add FKGL evaluation; using https://github.com/mmautner/readability
    -- 29th Feb: A whole pipeline from loading model to get the final evaluation results and analysis file

==v2.6.6==
1. add word embedding support (should be fast to implement)
2. add newela preprocessing code


==v2.6.10==
1. add EncDecAWE.lua -- fineTune works this time

==v2.6.12==
1. add recDropout option
2. add SGD fineTune after Adam

==v.2.6.13==
1. fix sampleA batch size 1
  a) change view in EncDecAWE and EncDecALN to support batchSize 1
  b) fix sampleA.lua to support batchSize = 1

==encdec-lm-v1.0==
forked from v2.6.13 (fixed batch size = 1)
1. add sampleALM.lua

==encdec-lm-tm-v1.0==
forked from encdec-lm-v1.0
1. add lex_trans.lua

==encdec-lm-tm-v1.1==
forked from encdec-lm-tm-v1.0
1. discounting on self-translation
  -- modify lex_trans.lua
  
==encdec-lm-neu-tm-1.0==
1. add neural aligner to generate training data for neural translation model
  -- add aligner.lua

2. add NeuLexTrans.lua -- this is a neural lexical translation model
  -- done!

3. change alignment
  -- a) hard alignment: one-best
  -- b) hard alignment: stochastic
  
==encdec-lm-neu-tm-encdec-rf-1.51.2==

3. change alignment
  -- a) hard alignment: one-best (done in v1.2)
  -- b) hard alignment: stochastic (done in v1.2)
  
  -- c) soft alignment

==encdec-lm-neu-tm-1.3==
  a) inegrate NTM into decoder 


==encdec-lm-neu-tm-dyn-wts-1.0==
1. design a gate to control how each feature weight
  -- control gate done!
  -- seems to improve PPL a little bit; further experiments for BLEU needed!
  
==encdec-lm-neu-tm-dyn-wts-1.1==
2. add feature weigencdec-rf-1.5hter in sample code.
  -- add sampleAWtLMTM.lua


==encdec-lm-neu-tm-dyn-wts-1.2==
3. more flexible model
  -- update NeuDynFeatWeighter.lua


==encdec-lm-neu-tm-dyn-wts-1.3==
add reinforce learning part
1. model done EncDecARF.lua
2. main loop done train_rf.lua (remove early stopping; save model per epoch; add --rfEpoch)


==encdec-lm-neu-tm-dyn-wts-1.4==
3. add validation show SARI


encdec-rf-1.5
==encdec-lm-neu-tm-dyn-wts-1.5==
3. show output after RF


==encdec-rf-1.0==
1. add language model as a reward


==encdec-rf-1.1==
1. implment a sequence auto-encoder


==encdec-rf-1.2==
1. add sequence auto-encoder as a similarity scorer


==encdec-rf-1.3==
1. add `symmetric SARI` metric
  -- modify SARI.lua


2. add relative weight for SARI and SARI-1


==encdec-rf-1.5==
forked from encdec-rf-1.3
1. better anonymize_ner.lua with larger dataset


==encdec-rf-1.6==
forked from encdec-rf-1.5
1. fix bug in NER replacement


