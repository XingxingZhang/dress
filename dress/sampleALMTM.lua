
require '.'
require 'shortcut'
require 'EncDec'
require 'EncDecA'
require 'EncDecAWE'
require 'EncDecALN'
require 'LSTMLM'
require 'lex_trans'
local model_utils = require 'model_utils'
require 'basic'

local XSampler = torch.class('EncDecASampler')

function XSampler:__init(modelPath, lmPath, lmWeight, lexTransPath, lexTransWeight, selfTransDiscount)
  assert( type(lmWeight) == 'number', 'LM weight must be a number' )
  self.lmWeight = lmWeight
  xprintln('LM Weight %f', lmWeight)
  
  local statePath = modelPath:sub(1, -3) .. 'state.t7'
  local opts = torch.load(statePath)
  
  self.lexTransWeight = lexTransWeight
  xprintln('Lex Trans Weight %f', lexTransWeight)
  xprintln('self Trans Discount %f', selfTransDiscount)
  self.lex_trans = LexTransModel(selfTransDiscount)
  self.lex_trans:load(opts.src_vocab, opts.dst_vocab, lexTransPath)
  
  local lmStatePath = lmPath:sub(1, -3) .. 'state.t7'
  local lm_opts = torch.load(lmStatePath)
  
  xprintln('opts dst_vocab EOS is %d, lm_opts vocab EOS is %d', opts.dst_vocab.EOS, lm_opts.vocab.EOS)
  assert(opts.dst_vocab.EOS == lm_opts.vocab.EOS, 'MUST have the same EOS')
  
  if opts.useGPU then
    require 'cunn'
    require 'cutorch'
  end
  torch.manualSeed(opts.seed)
  cutorch.manualSeed(opts.seed)
  print('load state done!')
  self:showOpts(opts)
  if opts.model == 'EncDec' then
    self.encdec = EncDec(opts)
  elseif opts.model == 'EncDecA' then
    self.encdec = EncDecA(opts)
  elseif opts.model == 'EncDecAWE' then
    self.encdec = EncDecAWE(opts)
  elseif opts.model == 'EncDecALN' then
    self.encdec = EncDecALN(opts)
  else
    error( string.format('model type [%s]! only support EncDec and EncDecA!\n', opts.model) )
  end
  self.encdec:load(modelPath)
  print('load model done!')
  self.opts = opts
  
  -- init model
  local bs = 1
  -- if opts.model == 'EncDecA' or opts.model == 'EncDecAWE' or opts.model == 'EncDecALN' then bs = 2 end
  self.s = {}
  for i = 1, 2*opts.nlayers do
    self.s[i] = self:transData( torch.zeros(bs, opts.nhid) )
  end
  print(self.s)
  self.h_hat = self:transData( torch.zeros(bs, opts.nhid) )
  self.all_enc_hs = self:transData( torch.zeros( bs * self.opts.seqLen, self.opts.nhid ) )
  print(self.all_enc_hs:size())
  
  -- get modules map (for attention module)
  self.encdec.modules_map = BModel.get_module_map({self.encdec.enc_lstm_master, self.encdec.dec_lstm_master})
  print 'get modules map done!'
  print(self.encdec.modules_map)
  
  -- create and init lstmlm
  self.lm_opts = lm_opts
  local tmp_vocab = lm_opts.vocab
  lm_opts.vocab = nil
  print(lm_opts)
  lm_opts.vocab = tmp_vocab
  self.lm = LSTMLM(lm_opts)
  self.lm:load(lmPath)
  
  self.lm_s = {}
  for i = 1, lm_opts.nlayers*2 do
    self.lm_s[i] = self:transData( torch.zeros(bs, lm_opts.nhid) )
  end
  print(self.lm_s[i])
end

function XSampler:transData(d)
  if self.opts.useGPU then 
    return d:cuda()
  else
    return d
  end
end

function XSampler:showOpts(opts)
  local tmp_src = opts.src_vocab
  local tmp_dst = opts.dst_vocab
  opts.src_vocab = nil
  opts.dst_vocab = nil
  print(opts)
  opts.src_vocab = tmp_src
  opts.dst_vocab = tmp_dst
end

function XSampler:generate(src_sent)
  local src_words = src_sent:splitc(' \t\r\n')
  local src_wids = {}
  table.insert(src_wids, self.opts.src_vocab.EOS)
  local function getwid(word)
    local wid = self.opts.src_vocab.word2idx[word]
    if wid == nil then wid = self.opts.src_vocab.UNK end
    return wid
  end
  
  for i = #src_words, 1, -1 do
    table.insert(src_wids, getwid(src_words[i]))
  end
  -- init s
  for i = 1, 2 * self.opts.nlayers do
    self.s[i]:zero()
  end
  -- encoder
  local Tx = #src_wids
  for t = 1, Tx do
    local x_t = self:transData( torch.Tensor({src_wids[t]}) )
    local nx_s = self.encdec.enc_lstm_master:forward({ x_t, self.s })
    model_utils.copy_table(self.s, nx_s)
  end
  
  local prev_word = self.opts.dst_vocab.EOS
  printf('first target word is %d : %s\n', prev_word, self.opts.dst_vocab.idx2word[prev_word] )
  local out_sent = {}
  -- currently we are doing beam search with beam size 1
  while true do
    -- move one step
    local x_t = self:transData( torch.Tensor({prev_word}) )
    local nx_s, predict, nx_h_hat = unpack( self.encdec.dec_lstm_master:forward({x_t, self.s}) )
    model_utils.copy_table(self.s, nx_s)
    local _, _prev_word = predict:max(2)
    prev_word = _prev_word[{1, 1}]
    -- print(self.opts.dst_vocab.idx2word[prev_word])
    
    if prev_word == self.opts.dst_vocab.EOS then 
      break
    else
      table.insert(out_sent, self.opts.dst_vocab.idx2word[prev_word])
    end
    
  end
  
  local out_text = ''
  for _, word in ipairs(out_sent) do
    out_text = out_text .. word .. ' '
  end
  
  return out_text:trim()
end

function XSampler:generate_att(src_sent)
  -- self.encdec.modules_map
  local src_words = src_sent:splitc(' \t\r\n')
  
  local src_wids = {}
  table.insert(src_wids, self.opts.src_vocab.EOS)
  local function getwid(word)
    local wid = self.opts.src_vocab.word2idx[word]
    if wid == nil then wid = self.opts.src_vocab.UNK end
    return wid
  end
  
  for i = #src_words, 1, -1 do
    table.insert(src_wids, getwid(src_words[i]))
  end
  
  for i = 1, 2 * self.opts.nlayers do
    self.s[i]:zero()
  end
  -- encoder
  local Tx = #src_wids
  self.all_enc_hs:resize( 1, Tx, self.opts.nhid )
  local raw_idxs = self:transData( torch.linspace(0, Tx*(2-1), 2):long() )
  for t = 1, Tx do
    local x_t = self:transData( torch.Tensor({src_wids[t]}) )
    local nx_s = self.encdec.enc_lstm_master:forward({ x_t, self.s })
    self.all_enc_hs[{ {}, t, {} }] = nx_s[2*self.opts.nlayers]
    model_utils.copy_table(self.s, nx_s)
  end
  local all_enc_hs = self.all_enc_hs
  
  -- get translation probs for each source word
  local trans_prob = self.lex_trans:lookup_word_index( torch.LongTensor(src_wids) )
  
  for i = 1, 2*self.lm_opts.nlayers do
    self.lm_s[i]:zero()
  end
  
  self.h_hat:zero()
  local x_mask_t = self:transData( torch.ones(1, Tx) )
  local x_mask_sub = self:transData( torch.zeros(1, Tx) )
  -- self.dec_hs_hat[t-1], all_enc_hs, x_mask_t, x_mask_sub
  local prev_word = self.opts.dst_vocab.EOS
  -- printf('first target word is %d : %s\n', prev_word, self.opts.dst_vocab.idx2word[prev_word] )
  local attention_scores = {}
  local out_sent = {}
  -- currently we are doing beam search with beam size 1
  local wd_cnt = 0
  while true do
    -- move one step
    local x_t = self:transData( torch.Tensor({prev_word}) )
    
    local nx_s, predict, nx_h_hat = unpack( self.encdec.dec_lstm_master:forward({x_t, self.s,
          self.h_hat, all_enc_hs, x_mask_t, x_mask_sub}) )
    model_utils.copy_table(self.s, nx_s)
    self.h_hat:copy(nx_h_hat)
    
    local nx_lm_s, lm_predict = unpack( self.lm.lstm_master:forward({x_t, self.lm_s}) )
    model_utils.copy_table(self.lm_s, nx_lm_s)
    
    wd_cnt= wd_cnt + 1
    --[[
    local dynamicWeight = wd_cnt / Tx * self.lmWeight
    if wd_cnt < Tx/3 then
      dynamicWeight = 0
    end
    -- interpolate
    predict = predict*(1-self.lmWeight) + lm_predict * dynamicWeight
    --]]
    local attention = self.encdec.modules_map.attention.output:float()
    table.insert(attention_scores, attention)
    
    if wd_cnt > 1 then
      local weigthed_trans_prob = torch.mm(attention[{ {}, {2, -1} }], trans_prob[{ {1, -2}, {} }])
      if self.opts.useGPU then weigthed_trans_prob = weigthed_trans_prob:cuda() end
      local predict_prob = torch.exp(predict) * (1 - self.lexTransWeight) + weigthed_trans_prob * self.lexTransWeight
      predict = torch.log( predict_prob )
    end
    
    local _, _prev_word = predict:max(2)
    prev_word = _prev_word[{1, 1}]
    
    -- show attention scores
    print('=attention scores:=')
    local sorted_att, att_idxs = torch.sort(attention, 2, true)
    for i = 1, sorted_att:size(2) do
      xprintln('%d %s %f', att_idxs[{1, i}], self.opts.src_vocab.idx2word[ src_wids[ att_idxs[{1, i}] ] ], sorted_att[{ 1, i }])
    end
    
    print('=predict word=')
    print(self.opts.dst_vocab.idx2word[prev_word])
    print ''
    
    if prev_word == self.opts.dst_vocab.EOS then 
      break
    else
      table.insert(out_sent, self.opts.dst_vocab.idx2word[prev_word])
    end
    
    if wd_cnt > 1.5 * #src_wids then break end
  end
  
  local out_text = ''
  for _, word in ipairs(out_sent) do
    out_text = out_text .. word .. ' '
  end
  
  return out_text:trim(), attention_scores
end

function XSampler:generateBatch(dataPath, outPath)
  self.encdec.enc_lstm_master:evaluate()
  self.encdec.dec_lstm_master:evaluate()
  local srcfin = io.open(dataPath .. '.src')
  local dstfin = io.open(dataPath .. '.dst')
  local cnt = 0
  local fout = io.open(outPath, 'w')
  local fout_log = io.open(outPath .. '.log', 'w')
  
  local attScores = {}
  while true do
    local sline = srcfin:read()
    local dline = dstfin:read()
    if sline == nil then
      assert(dline == nil)
      break
    end
    
    printf('Source Line:\n')
    printf('%s\n', sline)
    printf('Generate Line:\n')
    
    fout_log:write('Source Line:\n')
    fout_log:write(sline .. '\n')
    fout_log:write('Generate Line:\n')
    
    local generateLine, attScore
    if self.opts.model == 'EncDec' then
      generateLine = self:generate(sline)
    elseif self.opts.model == 'EncDecA' or self.opts.model == 'EncDecAWE' or self.opts.model == 'EncDecALN' then
      generateLine, attScore = self:generate_att(sline)
    else
      error('only support EncDec and EncDecA')
    end
    printf('%s\n', generateLine)
    printf('Reference Line:\n')
    printf('%s\n', dline)
    printf('\n\n\n')
    
    fout_log:write(generateLine .. '\n')
    fout_log:write('Reference Line:\n')
    fout_log:write(dline .. '\n')
    fout_log:write('\n\n\n')
    
    cnt = cnt + 1
    
    -- fout:write('Source Line:\n')
    -- fout:write(sline .. '\n')
    -- fout:write('Generate Line:\n')
    fout:write(generateLine .. '\n')
    -- fout:write('Reference Line:\n')
    -- fout:write(dline .. '\n')
    -- fout:write('\n\n')
    table.insert(attScores, attScore)
  end
  fout:close()
  fout_log:close()
  
  if #attScores > 0 then
    torch.save(outPath .. '.att.t7', attScores)
  end
end

local function main()
  local cmd = torch.CmdLine()
  
  cmd:text('Options:')
  cmd:option('--modelPath', '/disk/scratch/XingxingZhang/encdec/sent_simple/encdec_attention_PWKP_margin_prob/model_0.001.256.dot.2L.adam.reload.sgd.m0.97.t7',
    'model path')
  cmd:option('--dataPath', '/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/PWKP/an_ner/PWKP_108016.tag.80.aner.valid',
    'data path')
  cmd:option('--outPath', '/disk/scratch/XingxingZhang/encdec/sent_simple/encdec_attention_PWKP_margin_prob/sampleA/model_0.001.256.dot.2L.adam.reload.sgd.m0.97.valid',
    'output path')
  local opts = cmd:parse(arg)
  
  local sampler = EncDecASampler(opts.modelPath)
  sampler:generateBatch(opts.dataPath, opts.outPath)
end

if not package.loaded['sampleALMTM'] then
	main()
end



