
require '.'
require 'shortcut'
require 'paths'

require 'UniText_Dataset'
require 'EncDec'


local Trainer = torch.class('SeqAutoEncTrainer')


function Trainer:showOpts()
  local tmp_src = self.opts.src_vocab
  local tmp_dst = self.opts.dst_vocab
  self.opts.src_vocab = nil
  self.opts.dst_vocab = nil
  print(self.opts)
  self.opts.src_vocab = tmp_src
  self.opts.dst_vocab = tmp_dst
end


function Trainer:train()
  --[[
  local dataIter = UniText_Dataset.createBatch(self.opts.src_vocab, self.opts.dst_vocab, 
    self.opts.train, self.opts.batchSize)
  --]]
  local dataIter = UniText_Dataset.createBatchShuffle(self.opts.src_vocab, self.opts.dst_vocab, 
    self.opts.train, self.opts.batchSize)
  local dataSize, curDataSize = self.trainSize, 0
  local percent, inc = 0.001, 0.001
  local timer = torch.Timer()
  -- local sgdParam = {learningRate = opts.curLR}
  local sgdParam = self.opts.sgdParam
  local cnt = 0
  local totalLoss = 0
  local totalCnt = 0
  for x, x_mask, y in dataIter do
    local loss = self.encdec:trainBatch(x, x_mask, y, sgdParam)
    local y_mask = y[{{2, -1}, {}}]:ne(0)
    local nll = loss * x:size(2) / (y_mask:sum())
    
    totalLoss = totalLoss + loss * x:size(2)
    totalCnt = totalCnt + y_mask:sum()
    
    curDataSize = curDataSize + x:size(2)
    local ratio = curDataSize/dataSize
    if ratio >= percent then
      local wps = totalCnt / timer:time().real
      xprint( '\r%s %.3f %.4f (%s) / %.2f wps ... ', EPOCH_INFO, ratio, totalLoss/totalCnt, readableTime(timer:time().real), wps )
      percent = math.floor(ratio / inc) * inc
      percent = percent + inc
    end
    
    cnt = cnt + 1
    if cnt % 5 == 0 then
      collectgarbage()
    end
  end
  
  return totalLoss / totalCnt
end


function Trainer:valid(validFile)
  -- self.encdec:disableDropout()
  --[[
  local dataIter = UniText_Dataset.createBatch(self.opts.src_vocab, self.opts.dst_vocab, 
    self.opts.train, self.opts.batchSize, self.ncedata)
  --]]
  local dataIter = UniText_Dataset.createBatch(self.opts.src_vocab, self.opts.dst_vocab,
    validFile, self.opts.validBatchSize)
  local totalCnt = 0
  local totalLoss = 0
  local cnt = 0
  for x, x_mask, y in dataIter do
    local loss = self.encdec:validBatch(x, x_mask, y)
    totalLoss = totalLoss + loss * x:size(2)
    totalCnt = totalCnt + y[{ {2, -1}, {} }]:ne(0):sum()
    cnt = cnt + 1
    if cnt % 5 == 0 then
      collectgarbage()
    end
  end
  
  -- self.encdec:enableDropout()
  
  local entropy = totalLoss / totalCnt
  local ppl =  torch.exp(entropy)
  return {entropy = entropy, ppl = ppl}
end

function Trainer:mergeVocabs(opts, src_path, dst_path)
  -- add src_vocab and dst_vocab to opts
  local ori_src_vocab = torch.load(src_path)
  local ori_dst_vocab = torch.load(dst_path)
  local vocab = {
    src_freqCut = ori_src_vocab['freqCut'],
    src_ignoreCase = ori_src_vocab['ignoreCase'],
    dst_freqCut = ori_dst_vocab['freqCut'],
    dst_ignoreCase = ori_dst_vocab['ignoreCase']
  }
  
  local function show_vocab(vocab)
    for k, v in pairs(vocab) do
      if type(v) == 'table' then
        xprintln('%s -- table', k)
      else
        xprintln('%s -- %s', k, tostring(v))
      end
    end
  end
  
  xprintln('\n\n*** for src vocab ***')
  show_vocab(ori_src_vocab)
  
  xprintln('\n\n*** for dst vocab ***')
  show_vocab(ori_dst_vocab)
  
  local word2idx = {}
  local word_cnt = 0
  
  local function add_wd(wd)
    if word2idx[wd] == nil then
      word_cnt = word_cnt + 1
      word2idx[wd] = word_cnt
    end
  end
  
  for wid, wd in pairs(ori_src_vocab.idx2word) do
    if wid ~= ori_src_vocab.EOS then
      add_wd(wd)
    end
  end
  
  local EOS
  for wid, wd in pairs(ori_dst_vocab.idx2word) do
    add_wd(wd)
    if wid == ori_dst_vocab.EOS then
      xprintln('ori dst EOS %s, %d', wd, wid)
      EOS = word_cnt
    end
  end
  
  local idx2word = {}
  for k, v in pairs(word2idx) do
    idx2word[v] = k
  end
  
  xprintln('word cnt = %d, EOS = %s, %d, #word = %d', word_cnt, idx2word[EOS], EOS, #idx2word)
  
  vocab.word2idx = word2idx
  vocab.idx2word = idx2word
  vocab.UNK = 1
  vocab.EOS = EOS
  vocab.nvocab = word_cnt
  
  xprintln('\n\n*** final vocab ***')
  show_vocab(vocab)
  xprintln('merge vocabularies done!')
  
  opts.src_vocab = vocab
  opts.dst_vocab = vocab
  
  return opts
end


function Trainer:main()
  local model_opts = require 'auto_encoder_opts'
  local opts = model_opts.getOpts()
  self.opts = opts
  
  self:mergeVocabs(opts, opts.ori_src_vocab_path, opts.ori_dst_vocab_path)
  
  self.trainSize, self.validSize = unpack( UniText_Dataset.getDataSize({opts.train, opts.valid}) )
  printf('train size = %d, valid size = %d\n', self.trainSize, self.validSize)
  
  if opts.model == 'EncDec' then
    self.encdec = EncDec(opts)
  else
    error('only support auto encoder!')
  end
  self:showOpts()
  local encdec = self.encdec
  
  local bestValid = {ppl = 1e309, entropy = 1e309}
  local lastValid = {ppl = 1e309, entropy = 1e309}
  local bestModel = torch.FloatTensor(self.encdec.params:size())
  local patience = opts.patience
  local divLR = false
  local timer = torch.Timer()
  local epochNo = 0
  for epoch = 1, opts.maxEpoch do
    epochNo = epochNo + 1
    EPOCH_INFO = string.format('epoch %d', epoch)
    local startTime = timer:time().real
    local trainCost = self:train()
    -- print('training ignored!!!')
    -- local trainCost = 123
    xprint('\repoch %d TRAIN nll %f ', epoch, trainCost)
    local validRval = self:valid(opts.valid)
    xprint('VALID %f ', validRval.ppl)
    -- local testRval = valid(rnn, lmdata, opts, 'test')
    -- xprint('TEST %f ', testRval.ppl)
    local endTime = timer:time().real
    xprintln('lr = %.4g (%s) p = %d', opts.curLR, readableTime(endTime - startTime), patience)
    -- self:validBLEU(opts.valid, string.format('%s.e%d', opts.validout, epoch))
    -- self:validBLEU(opts.test, string.format('%s.e%d', opts.testout, epoch))
    
    local oldBestValid_PPL = bestValid.ppl
    if validRval.ppl < bestValid.ppl then
      bestValid.ppl = validRval.ppl
      bestValid.entropy = validRval.entropy
      bestValid.epoch = epoch
      encdec:getModel(bestModel)
      -- for non SGD algorithm, we will reset the patience
      -- if opts.optimMethod ~= 'SGD' then
      if opts.lrDiv <= 1 then
        patience = opts.patience
      end
    end
    
    -- if validRval.ppl + 1 > oldBestValid_PPL then
    if validRval.ppl > oldBestValid_PPL then
      printf('valid ppl %f\n', validRval.ppl)
      printf('best ppl %f\n', oldBestValid_PPL)
      -- non SGD algorithm decrease patience
      if opts.lrDiv <= 1 then
      -- if opts.optimMethod ~= 'SGD' then
        patience = patience - 1
        if patience == 0 then
          xprintln('No improvement on PPL for %d epoch(s). Training finished!', opts.patience)
          break
        end
      else
        -- SGD with learning rate decay
        encdec:setModel(bestModel)
      end
    end -- if validRval.ppl + 1 > bestValid.ppl
    
    if opts.savePerEpoch then
      local tmpPath = opts.save:sub(1, -4) .. '.tmp.t7'
      encdec:save(tmpPath, true)
    end
    
    if opts.saveBeforeLrDiv then
      if opts.optimMethod == 'SGD' and opts.curLR == opts.lr then
        local tmpPath = opts.save:sub(1, -4) .. '.blrd.t7'
        encdec:save(tmpPath, true)
      end
    end
    
    -- control the learning rate decay
    -- if opts.optimMethod == 'SGD' then
    if opts.lrDiv > 1 then
      if epoch >= 10 and patience > 1 then
        patience = 1
      end
      
      if validRval.entropy * opts.minImprovement > lastValid.entropy then
        if not divLR then  -- patience == 1
          patience = patience - 1
          if patience < 1 then divLR = true end
        else
          xprintln('no significant improvement! cur ppl %f, best ppl %f', validRval.ppl, bestValid.ppl)
          break
        end
      end
      
      if divLR then
        opts.curLR = opts.curLR / opts.lrDiv
        opts.sgdParam.learningRate = opts.curLR
      end
      
      if opts.curLR < opts.minLR then
        xprintln('min lr is met! cur lr %e min lr %e', opts.curLR, opts.minLR)
        break
      end
      lastValid.ppl = validRval.ppl
      lastValid.entropy = validRval.entropy
    end
  end
  
  if epochNo >= opts.maxEpoch then
    xprintln('Max number of epoch is met. Training finished!')
  end
  
  encdec:setModel(bestModel)
  opts.sgdParam = nil
  encdec:save(opts.save, true)
  xprintln('model saved at %s', opts.save)
  
  local validRval = self:valid(opts.valid)
  xprint('VALID %f \n', validRval.ppl)
  
  if opts.test and opts.test ~= '' then
    local testRval = self:valid(opts.test)
    xprint('TEST %f \n', testRval.ppl)
  end
end


local function main()
  local trainer = SeqAutoEncTrainer()
  trainer:main()
end


if not package.loaded['train_auto_encoder'] then
  main()
end