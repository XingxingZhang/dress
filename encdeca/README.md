
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
