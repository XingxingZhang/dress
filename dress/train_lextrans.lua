
require '.'
require 'NeuLexTrans'
require 'NeuLexTransSoft'
require 'LexTrans_Dataset'

local function getOpts()
  local cmd = torch.CmdLine()
  cmd:option('--srcVocab', '', 'vocab path')
  cmd:option('--dstVocab', '', 'vocab path')
  cmd:option('--dataset', '', 'dataset path')
  
  cmd:option('--seed', 123, '')
  cmd:option('--initRange', 0.1, '')
  cmd:option('--nhid', 256, '')
  cmd:option('--nin', 128, '')
  cmd:option('--nlayers', 2, '')
  cmd:option('--dropout', 0, 'dropout')
  cmd:option('--useGPU', false, '')
  cmd:option('--seqLen', 85, '')
  cmd:option('--optimMethod', 'Adam', '')
  cmd:option('--lr', 0.001, '')
  cmd:option('--batchSize', 32, '')
  cmd:option('--maxEpoch', 30, '')
  cmd:option('--wordEmbedding', '', '')
  cmd:option('--embedOption', 'init', 'init fineTune')
  cmd:option('--fineTuneFactor', 0, '')
  cmd:option('--decStart', 1, 'start pos in decoder')
  cmd:option('--patience', 1, 'patience')
  cmd:option('--lrDiv', 0, 'learning rate decay when there is no significant improvement. 0 means turn off')
  
  cmd:option('--save', 'model.t7', '')
  
  cmd:option('--gradClip', 5, 'grad clip: > 0 for rescale norm; < 0 for clip')
  
  cmd:option('--alignOption', 'hard-one-best', 'options: hard-one-best, hard-stochastic, soft')
  
  local opts = cmd:parse(arg)
  torch.manualSeed(opts.seed)
  if opts.useGPU then
    require 'cutorch'
    require 'cunn'
    cutorch.manualSeed(opts.seed)
  end
  
  opts.curLR = opts.lr
  opts.sgdParam = {learningRate = opts.lr}
  if opts.optimMethod == 'AdaDelta' then
    opts.rho = 0.95
    opts.eps = 1e-6
    opts.sgdParam.rho = opts.rho
    opts.sgdParam.eps = opts.eps
  elseif opts.optimMethod == 'SGD' then
    if opts.lrDiv <= 1 then
      opts.lrDiv = 2
    end
  end
  
  local validAlignOptions = {'hard-one-best', 'hard-stochastic', 'soft'}
  if not table.contains(validAlignOptions, opts.alignOption) then
    error('invalid align option! ' .. opts.alignOption)
  end
  
  print(opts)
  
  return opts
end

local Trainer = torch.class('NeuLexTransTrainer')

function Trainer:train()
  local dataIter
  if self.opts.alignOption == 'hard-one-best' then
    dataIter = LexTrans_Dataset.createBatch(self.dataset.train, self.opts.batchSize, self.opts.UNK, self.opts.DEF_POS, true)
  elseif self.opts.alignOption == 'hard-stochastic' then
    dataIter = LexTrans_Dataset.createBatchStochastic(self.dataset.train, self.opts.batchSize, self.opts.UNK, self.opts.DEF_POS, true)
  end
  
  if self.opts.alignOption == 'soft' then
    dataIter = LexTrans_Dataset.createBatchSoft(self.dataset.train, self.opts.batchSize, self.opts.UNK, self.opts.DEF_POS, true)
  end
  
  local dataSize, curDataSize = #self.dataset.train, 0
  local percent, inc = 0.001, 0.001
  local timer = torch.Timer()
  local sgdParam = self.opts.sgdParam
  local cnt = 0
  local totalLoss = 0
  local totalCnt = 0
  for x, x_mask, a, y in dataIter do
    local loss = self.net:trainBatch(x, x_mask, a, y, sgdParam)
    local y_mask = y[{{1 + self.opts.decStart, -1}, {}}]:ne(0)
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

function Trainer:valid(validSet)
  local dataIter = LexTrans_Dataset.createBatch(validSet, self.opts.batchSize, self.opts.UNK, self.opts.DEF_POS)
  if self.opts.alignOption == 'soft' then
    -- dataIter = LexTrans_Dataset.createBatchSoft(self.dataset.train, self.opts.batchSize, self.opts.UNK, self.opts.DEF_POS)
    dataIter = LexTrans_Dataset.createBatchSoft(validSet, self.opts.batchSize, self.opts.UNK, self.opts.DEF_POS)
  end
  local totalCnt = 0
  local totalLoss = 0
  local cnt = 0
  for x, x_mask, a, y in dataIter do
    local loss = self.net:validBatch(x, x_mask, a, y)
    totalLoss = totalLoss + loss * x:size(2)
    totalCnt = totalCnt + y[{ {1 + self.opts.decStart, -1}, {} }]:ne(0):sum()
    cnt = cnt + 1
    if cnt % 5 == 0 then
      collectgarbage()
    end
  end
  
  local entropy = totalLoss / totalCnt
  local ppl =  torch.exp(entropy)
  return {entropy = entropy, ppl = ppl}
end

function Trainer:main()
  local opts = getOpts()
  opts.src_vocab = torch.load(opts.srcVocab)
  opts.dst_vocab = torch.load(opts.dstVocab)
  if opts.alignOption == 'soft' then
    self.net = NeuLexTransSoft(opts)
  else
    self.net = NeuLexTrans(opts)
  end
  -- self.net = NeuLexTrans(opts)
  self.opts = opts
  
  local dataset_path = opts.dataset .. '.thin.detailed.t7'
  if paths.filep(dataset_path) then
    self.dataset = torch.load(dataset_path)
  else
    self.dataset = LexTrans_Dataset.createDataset(opts.src_vocab, opts.dst_vocab, opts.dataset)
    torch.save(dataset_path, self.dataset)
  end
  xprintln('#train = %d, #valid = %d, #test = %d', #self.dataset.train, #self.dataset.valid, #self.dataset.test)
  opts.DEF_POS = 1 -- default pos for alignment
  opts.UNK = opts.src_vocab.UNK
  
  local bestValid = {ppl = 1e309, entropy = 1e309}
  local lastValid = {ppl = 1e309, entropy = 1e309}
  local bestModel = torch.FloatTensor(self.net.params:size())
  local patience = opts.patience
  local divLR = false
  local timer = torch.Timer()
  local epochNo = 0
  for epoch = 1, opts.maxEpoch do
    epochNo = epoch
    timer:reset()
    EPOCH_INFO = string.format('epoch %d', epoch)
    local trainCost = self:train()
    -- print('training ignored!!!')
    -- local trainCost = 123
    xprint('\repoch %d TRAIN nll %f ', epoch, trainCost)
    local validRval = self:valid(self.dataset.valid)
    xprint('VALID %f ', validRval.ppl)
    xprintln('lr = %.4g (%s) p = %d', opts.curLR, readableTime(timer:time().real), patience)
    
    if validRval.ppl < bestValid.ppl then
      bestValid.ppl = validRval.ppl
      bestValid.entropy = validRval.entropy
      bestValid.epoch = epoch
      self.net:getModel(bestModel)
      -- for non SGD algorithm, we will reset the patience
      -- if opts.optimMethod ~= 'SGD' then
      if opts.lrDiv <= 1 then
        patience = opts.patience
      end
    end
    
    --[[
    local oldBestValid_PPL = bestValid.ppl
    if validRval.ppl < bestValid.ppl then
      bestValid.ppl = validRval.ppl
      bestValid.entropy = validRval.entropy
      bestValid.epoch = epoch
      self.net:getModel(bestModel)
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
        self.net:setModel(bestModel)
      end
    end -- if validRval.ppl + 1 > bestValid.ppl
    
    if opts.savePerEpoch then
      local tmpPath = opts.save:sub(1, -4) .. '.tmp.t7'
      self.net:save(tmpPath, true)
    end
    
    if opts.saveBeforeLrDiv then
      if opts.optimMethod == 'SGD' and opts.curLR == opts.lr then
        local tmpPath = opts.save:sub(1, -4) .. '.blrd.t7'
        self.net:save(tmpPath, true)
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
    --]]
  end
  
  self.net:setModel(bestModel)
  opts.sgdParam = nil
  self.net:save(opts.save, true)
  xprintln('model saved at %s', opts.save)
  
  local validRval = self:valid(self.dataset.valid)
  xprint('VALID %f \n', validRval.ppl)
  
  local testRval = self:valid(self.dataset.test)
  xprint('TEST %f \n', testRval.ppl)
end

local function main()
  local trainer = NeuLexTransTrainer()
  trainer:main()
end

if not package.loaded['train_lextrans'] then
  main()
end

