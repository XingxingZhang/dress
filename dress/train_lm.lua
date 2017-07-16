
require '.'
require 'shortcut'
require 'paths'

require 'LSTMLM'
-- require 'RawLM_Dataset'
require 'RawLM_Dataset_Legacy'

local model_utils = require 'model_utils'
local EPOCH_INFO = ''

local lmdata = RawLM_Dataset()

local function getOpts()
  local cmd = torch.CmdLine()
  cmd:text('====== LSTM Language Model ======')
  cmd:text('add early stopping')
  cmd:option('--seed', 123, 'random seed')
  cmd:option('--model', 'LSTMLM', 'model options: currently only support LSTMLM')
  cmd:option('--train', '/disk/scratch/XingxingZhang/lm/lstm_/data/ptb.train.txt', 'train file')
  cmd:option('--freqCut', 0, 'for word frequencies')
  cmd:option('--ignoreCase', false, 'whether you will ignore the case')
  cmd:option('--valid', '/disk/scratch/XingxingZhang/lm/lstm_/data/ptb.valid.txt', 'valid file')
  cmd:option('--test', '/disk/scratch/XingxingZhang/lm/lstm_/data/ptb.test.txt', 'test file (in default: no test file)')
  -- cmd:option('--vocab', '/disk/scratch1/xingxing.zhang/xTreeRNN/dataset/msr/msr.100.sort.vocab.tmp.t7', 'valid file')
  -- cmd:option('--dataset', '', 'dataset path')
  cmd:option('--maxEpoch', 20, 'maximum number of epochs')
  cmd:option('--batchSize', 64, '')
  cmd:option('--validBatchSize', 64, '')
  cmd:option('--nin', 100, 'word embedding size')
  cmd:option('--nhid', 200, 'hidden unit size')
  cmd:option('--window', 5, 'n-gram window size')
  cmd:option('--nlayers', 1, 'number of hidden layers')
  cmd:option('--shareEmbed', false, 'share the embeddings of LSTM and NNLM')
  cmd:option('--combine', 'GRU', 'the way to combine different models: linear combination and GRU')
  cmd:option('--lr', 0.1, 'learning rate')
  cmd:option('--lrDiv', 0, 'learning rate decay when there is no significant improvement. 0 means turn off')
  cmd:option('--minImprovement', 1.0001, 'if improvement on log likelihood is smaller then patient --')
  cmd:option('--optimMethod', 'AdaGrad', 'optimization algorithm')
  cmd:option('--gradClip', 5, '> 0 means to do Pascanu et al.\'s grad norm rescale http://arxiv.org/pdf/1502.04623.pdf; < 0 means to truncate the gradient larger than gradClip; 0 means turn off gradient clip')
  cmd:option('--initRange', 0.1, 'init range')
  cmd:option('--initHidVal', 0.01, 'init values for hidden states')
  cmd:option('--seqLen', 101, 'maximum seqence length')
  cmd:option('--useGPU', false, 'use GPU')
  cmd:option('--patience', 1, 'stop training if no lower valid PPL is observed in [patience] consecutive epoch(s)')
  cmd:option('--save', 'model.t7', 'save model path')
  
  cmd:text()
  cmd:text('Options for NCE')
  cmd:option('--nneg', 20, 'number of negative samples')
  cmd:option('--power', 0.75, 'for power for unigram frequency')
  cmd:option('--lnZ', 9, 'default normalization term')
  cmd:option('--learnZ', false, 'learn the normalization constant Z')
  cmd:option('--normalizeUNK', false, 'if normalize UNK or not')
  
  cmd:text()
  cmd:text('Options for long jobs')
  cmd:option('--savePerEpoch', false, 'save model every epoch')
  cmd:option('--saveBeforeLrDiv', false, 'save model before lr div')
  
  cmd:text()
  cmd:text('Options for regularization')
  cmd:option('--dropout', 0, 'dropout rate (dropping)')
  
  return cmd:parse(arg)
end

local function initOpts(opts)
  -- for different models
  local nceParams = {'nneg', 'power', 'normalizeUNK', 'learnZ', 'lnZ'}
  if opts.model == 'FFLSTMLM' then
    -- delete nce params
    for _, nceparam in ipairs(nceParams) do
      opts[nceparam] = nil
    end
  end
  
  -- for different optimization algorithms
  local optimMethods = {'AdaGrad', 'Adam', 'AdaDelta', 'SGD'}
  if not table.contains(optimMethods, opts.optimMethod) then
    error('invalid optimization problem ' .. opts.optimMethod)
  end
  
  opts.curLR = opts.lr
  opts.minLR = 1e-7
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
  
  torch.manualSeed(opts.seed)
  if opts.useGPU then
    require 'cutorch'
    require 'cunn'
    cutorch.manualSeed(opts.seed)
  end
end

local Trainer = torch.class('LSTMTrainer')

function Trainer:showOpts()
  local tmp = self.opts.vocab
  self.opts.vocab = nil
  print(self.opts)
  self.opts.vocab = tmp
end

function Trainer:train()
  local dataIter = RawLM_Dataset.createBatchShuffle(self.opts.vocab, self.opts.train, self.opts.batchSize)
  local dataSize, curDataSize = self.trainSize, 0
  local percent, inc = 0.001, 0.001
  local timer = torch.Timer()
  -- local sgdParam = {learningRate = opts.curLR}
  local sgdParam = self.opts.sgdParam
  local cnt = 0
  local totalLoss = 0
  local totalCnt = 0
  for x, y, y_neg, y_prob, y_neg_prob, mask in dataIter do
    local loss
    if y_neg then
      loss = self.rnn:trainBatch(x, y, y_neg, y_prob, y_neg_prob, mask, sgdParam)
    else
      loss = self.rnn:trainBatch(x, y, sgdParam)
    end
    
    local nll = loss * x:size(2) / (y:ne(0):sum())
    if mask then
      nll = loss * x:size(2) / (mask:sum())
    else
      nll = loss * x:size(2) / (y:ne(0):sum())
    end
    
    totalLoss = totalLoss + loss * x:size(2)
    if mask then
      totalCnt = totalCnt + mask:sum()
    else
      totalCnt = totalCnt + y:ne(0):sum()
    end
    
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
  local dataIter = RawLM_Dataset.createBatch(self.opts.vocab, validFile, self.opts.validBatchSize)
  local totalCnt = 0
  local totalLoss = 0
  local cnt = 0
  for x, y in dataIter do
    local loss = self.rnn:validBatch(x, y)
    totalLoss = totalLoss + loss * x:size(2)
    totalCnt = totalCnt + y:ne(0):sum()
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
  initOpts(opts)
  self.opts = opts
  local vocabPath = opts.train .. '.vocab.tmp.t7'
  if paths.filep(vocabPath) then
    opts.vocab = torch.load(vocabPath)
    print(opts.vocab.nvocab)
    print 'load vocab done!'
  else
    opts.vocab = RawLM_Dataset.createVocab(opts.train, opts.freqCut, opts.ignoreCase, true)
    print 'creat vocab done!'
    torch.save(vocabPath, opts.vocab)
    print 'save vocab done!'
  end
  opts.nvocab = opts.vocab.nvocab
  self.trainSize, self.validSize = unpack( RawLM_Dataset.getDataSize({opts.train, opts.valid}) )
  printf('train size = %d, valid size = %d\n', self.trainSize, self.validSize)
  
  self:showOpts()
  if opts.model == 'LSTMLM' then
    self.rnn = LSTMLM(opts)
  else
    error('invalid model option')
  end
  local rnn = self.rnn
  
  local bestValid = {ppl = 1e309, entropy = 1e309}
  local lastValid = {ppl = 1e309, entropy = 1e309}
  local bestModel = torch.FloatTensor(self.rnn.params:size())
  local patience = opts.patience
  local divLR = false
  local timer = torch.Timer()
  local epochNo = 0
  for epoch = 1, opts.maxEpoch do
    epochNo = epochNo + 1
    EPOCH_INFO = string.format('epoch %d', epoch)
    local startTime = timer:time().real
    -- local trainCost = train(rnn, lmdata, opts)
    local trainCost = self:train()
    -- print('training ignored!!!')
    -- local trainCost = 123
    xprint('\repoch %d TRAIN nll %f ', epoch, trainCost)
    local validRval = self:valid(opts.valid)
    xprint('VALID %f ', validRval.ppl)
    -- local testRval = valid(rnn, lmdata, opts, 'test')
    local testRval = self:valid(opts.test)
    xprint('TEST %f ', testRval.ppl)
    local endTime = timer:time().real
    xprintln('lr = %.4g (%s) p = %d', opts.curLR, readableTime(endTime - startTime), patience)
    
    local oldBestValid_PPL = bestValid.ppl
    if validRval.ppl < bestValid.ppl then
      bestValid.ppl = validRval.ppl
      bestValid.entropy = validRval.entropy
      bestValid.epoch = epoch
      rnn:getModel(bestModel)
      -- for non SGD algorithm, we will reset the patience
      -- if opts.optimMethod ~= 'SGD' then
      if opts.lrDiv <= 1 then
        patience = opts.patience
      end
    end
    
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
        rnn:setModel(bestModel)
      end
    end -- if validRval.ppl + 1 > bestValid.ppl
    
    if opts.savePerEpoch then
      local tmpPath = opts.save:sub(1, -4) .. '.tmp.t7'
      rnn:save(tmpPath, true)
    end
    
    if opts.saveBeforeLrDiv then
      if opts.optimMethod == 'SGD' and opts.curLR == opts.lr then
        local tmpPath = opts.save:sub(1, -4) .. '.blrd.t7'
        rnn:save(tmpPath, true)
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
  
  if epochNo > opts.maxEpoch then
    xprintln('Max number of epoch is met. Training finished!')
  end
  
  rnn:setModel(bestModel)
  opts.sgdParam = nil
  rnn:save(opts.save, true)
  xprintln('model saved at %s', opts.save)
  
  local validRval = self:valid(opts.valid)
  xprint('VALID %f \n', validRval.ppl)
  
  if opts.test and opts.test ~= '' then
    local testRval = self:valid(opts.test)
    xprint('TEST %f \n', testRval.ppl)
  end
end


local function main()
  local trainer = LSTMTrainer()
  trainer:main()
end

main()

