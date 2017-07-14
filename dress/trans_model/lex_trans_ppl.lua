
include '../utils/shortcut.lua'
require 'lex_trans'

local function computePPL(opts)
  local lex_trans = LexTransModel()
  lex_trans:load(opts.srcVocab, opts.dstVocab, opts.lexFile)
  local dataset = torch.load(opts.dataset)
  
  local function showSent(vocab, x, reverse)
    local i, iend, inc
    if reverse then
      i = x:size(1)
      iend = 1
      inc = -1
    else
      i = 1
      iend = x:size(1)
      inc = 1
    end
    local words = {}
    while true do
      words[#words + 1] = vocab.idx2word[x[i]]
      if i == iend then break end
      i = i + inc
    end
    return table.concat(words, ' ')
  end
  
  local function get_ppl(split, decStart)
    local smoothing = 1e-10
    local logprob, cnt = 0, 0
    local nzlogprob, nzcnt = 0, 0
    for _, xay in ipairs(split) do
      for j = decStart, xay.align:size(1) do
        local a = xay.align[j]
        local prob
        if a > 1 then
          --[[
          local src = lex_trans.src_vocab.idx2word[ xay.src[a] ]
          local src2 = lex_trans.src_vocab.idx2word[ xay.src[a-1] ]
          local dst = lex_trans.dst_vocab.idx2word[ xay.dst[j+1] ]
          xprintln('src = %s, src2 = %s, dst = %s', src, src2, dst)
          --]]
          local src_id = xay.src[a-1]
          local dst_id = xay.dst[j+1]
          local probs = lex_trans:lookup_word_index(torch.LongTensor({src_id}))
          prob = probs[{ 1, dst_id }]
          if prob <= smoothing then
            if prob ~= 0 then
              print(prob)
            end
            
            prob = smoothing
          end
        else
          prob = smoothing
        end
        
        logprob = logprob + math.log(prob)
        cnt = cnt + 1
        if prob ~= smoothing then
          nzcnt = nzcnt + 1
          nzlogprob = nzlogprob + math.log(prob)
        end
      end
    end
    
    xprintln('cnt = %d, nzcnt = %d', cnt, nzcnt)
    
    return math.exp(-logprob/cnt), math.exp(-nzlogprob/nzcnt)
    
    --[[
    print(#split)
    for i, xay in ipairs(split) do
      xprintln('x = %s', showSent(lex_trans.src_vocab, xay.src, true))
      xprintln('y = %s', showSent(lex_trans.dst_vocab, xay.dst, false))
      print '\n'
      
      print(xay.align)
      for j = 1, xay.align:size(1) do
        local a = xay.align[j]
        if a > 1 then
          local src = lex_trans.src_vocab.idx2word[ xay.src[a] ]
          local src2 = lex_trans.src_vocab.idx2word[ xay.src[a-1] ]
          local dst = lex_trans.dst_vocab.idx2word[ xay.dst[j+1] ]
          xprintln('src = %s, src2 = %s, dst = %s', src, src2, dst)
        end
      end
    end
    --]]
  end
  
  xprintln( 'valid ppl %f, nz ppl %f', get_ppl(dataset.valid, 2) )
  xprintln( 'test ppl %f, nz ppl %f', get_ppl(dataset.test, 2) )
  xprintln( 'train ppl %f, nz ppl %f', get_ppl(dataset.train, 2) )
end

local function getOpts()
  local cmd = torch.CmdLine()
  cmd:option('--srcVocab', '/disk/scratch1/xingxing.zhang/seq2seq/sent_simple/encdec_lm_neu_tm/V0V4_V1V4_V2V4_V3V4_V0V3_V0V2_V1V3.aner.train.src.vocab.tmp.t7', '')
  cmd:option('--dstVocab', '/disk/scratch1/xingxing.zhang/seq2seq/sent_simple/encdec_lm_neu_tm/V0V4_V1V4_V2V4_V3V4_V0V3_V0V2_V1V3.aner.train.dst.vocab.tmp.t7', '')
  cmd:option('--lexFile', '/disk/scratch1/xingxing.zhang/seq2seq/sent_simple/encdec_lm_tm/encdec_tm/ner.lex.f2e', '')
  cmd:option('--dataset', '/disk/scratch1/xingxing.zhang/seq2seq/sent_simple/encdec_lm_neu_tm/align/newsela.align.t7.thin.t7', '')
  
  local opts = cmd:parse(arg)
  print(opts)
  
  return opts
end

local function main()
  local opts = getOpts()
  computePPL(opts)
end

if not package.loaded['lex_trans_ppl'] then
  main()
end

