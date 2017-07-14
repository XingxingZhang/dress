
require '.'
require 'shortcut'
require 'EncDec'
require 'EncDecA'
require 'EncDecAWE'
require 'EncDecALN'
local model_utils = require 'model_utils'
require 'basic'

local XSampler = torch.class('EncDecASampler')

function XSampler:__init(modelPath)
  local statePath = modelPath:sub(1, -3) .. 'state.t7'
  local opts = torch.load(statePath)
  
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
    
    -- print 'attention scores'
    -- print(self.encdec.modules_map.attention.output)
    table.insert(attention_scores, self.encdec.modules_map.attention.output:float())
    
    local _, _prev_word = predict:max(2)
    prev_word = _prev_word[{1, 1}]
    
    -- print('predict word')
    -- print(self.opts.dst_vocab.idx2word[prev_word])
    
    if prev_word == self.opts.dst_vocab.EOS then 
      break
    else
      table.insert(out_sent, self.opts.dst_vocab.idx2word[prev_word])
    end
    
    wd_cnt= wd_cnt + 1
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

if not package.loaded['sampleA'] then
	main()
end


