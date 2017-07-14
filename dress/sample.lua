
require '.'
require 'shortcut'
require 'EncDec'
local model_utils = require 'model_utils'

local XSampler = torch.class('EncDecSampler')

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
  self.encdec = EncDec(opts)
  self.encdec:load(modelPath)
  print('load model done!')
  self.opts = opts
  
  self.s = {}
  for i = 1, 2*opts.nlayers do
    self.s[i] = self:transData( torch.zeros(1, opts.nhid) )
  end
  print(self.s)
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
  -- print(src_words)
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
  -- print(src_wids)
  -- init ds
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
  
  -- encoding results are in self.s
  --[[
  for i = 1, 2 * self.opts.nlayers do
    print(self.s[i])
  end
  --]]
  
  local prev_word = self.opts.dst_vocab.EOS
  printf('first target word is %d : %s\n', prev_word, self.opts.dst_vocab.idx2word[prev_word] )
  local out_sent = {}
  -- currently we are doing beam search with beam size 1
  while true do
    -- move one step
    local x_t = self:transData( torch.Tensor({prev_word}) )
    local nx_s, predict = unpack( self.encdec.dec_lstm_master:forward({x_t, self.s}) )
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

function XSampler:generateBatch(dataPath, outPath)
  self.encdec.enc_lstm_master:evaluate()
  self.encdec.dec_lstm_master:evaluate()
  local srcfin = io.open(dataPath .. '.src')
  local dstfin = io.open(dataPath .. '.dst')
  local cnt = 0
  local fout = io.open(outPath, 'w')
  
  while true do
    local sline = srcfin:read()
    local dline = dstfin:read()
    if sline == nil then
      assert(dline == nil)
      break
    end
    
    print('Source Line:')
    print(sline)
    print('Generate Line:')
    local generateLine = self:generate(sline)
    print(generateLine)
    print('Reference Line:')
    print(dline)
    print '\n\n\n'
    cnt = cnt + 1
    -- if cnt == 5 then break end
    fout:write('Source Line:\n')
    fout:write(sline .. '\n')
    fout:write('Generate Line:\n')
    fout:write(generateLine .. '\n')
    fout:write('Reference Line:\n')
    fout:write(dline .. '\n')
    fout:write('\n\n')
  end
  
  fout:close()
end

local function main()
  --[[
  local modelPath = '/disk/scratch1/xingxing.zhang/seq2seq/sent_simple/experiments/test_atis_debug/model_0.05.50.sp.t7'
  local sampler = EncDecSampler(modelPath)
  local dataPath = '/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/atis/dev.txt'
  local outPath = '/disk/scratch1/xingxing.zhang/seq2seq/sent_simple/experiments/test_atis_debug/haha.out.txt'
  --]]
  local cmd = torch.CmdLine()
  
  cmd:text('Options:')
  cmd:option('--modelPath', '/disk/scratch1/xingxing.zhang/seq2seq/sent_simple/experiments/test_atis_debug/model_0.05.50.sp.t7',
    'model path')
  cmd:option('--dataPath', '/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/atis/dev.txt',
    'data path')
  cmd:option('--outPath', '/disk/scratch1/xingxing.zhang/seq2seq/sent_simple/experiments/test_atis_debug/haha.out.txt',
    'output path')
  local opts = cmd:parse(arg)
  local sampler = EncDecSampler(opts.modelPath)
  sampler:generateBatch(opts.dataPath, opts.outPath)
end

main()

