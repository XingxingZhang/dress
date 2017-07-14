
require '.'
require 'EncDecAWE'
require 'NeuLexTransSoft'
require 'NeuDynFeatWeighter'
require 'BiText_Dataset'


local function getOpts()
  local cmd = torch.CmdLine()
  cmd:option('--encdec_path', '', '')
  cmd:option('--neutrans_path', '', '')
  
  cmd:option('--seed', 123, '')
  cmd:option('--initRange', 0.001, '')
  cmd:option('--useGPU', false, '')
  
  cmd:option('--batchSize', 32, '')
  cmd:option('--nfeat', 2, 'number of neural features')
  cmd:option('--nhids', '256,256', '')
  cmd:option('--nghid', 256, '')
  cmd:option('--dropout', 0, '')
  cmd:option('--inDropout', 0, '')
  cmd:option('--maxEpoch', 10, '')
  
  cmd:option('--optimMethod', 'Adam', '')
  cmd:option('--lr', 0.001, '')
  cmd:option('--patience', 1, 'patience')
  cmd:option('--lrDiv', 0, 'learning rate decay when there is no significant improvement. 0 means turn off')
  
  cmd:option('--train', '', '')
  cmd:option('--valid', '', '')
  cmd:option('--test', '', '')
  
  cmd:option('--deepGate', false, 'use deep softmax gate')
  
  cmd:option('--save', 'model.t7', '')
  
  local opts = cmd:parse(arg)
  -- convert string `nhids` to table
  local nhids_s = opts.nhids:splitc(',')
  opts.nhids = {}
  for _, hid_s in ipairs(nhids_s) do
    table.insert(opts.nhids, tonumber(hid_s))
  end
  
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
  
  return opts
end

local Trainer = torch.class('NeuDynFeatTrainer')

function Trainer:train()
  local dataIter = BiText_Dataset.createBatchShuffle(self.src_vocab, self.dst_vocab, 
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
    local loss = self.weighter:trainBatch(x, x_mask, y, sgdParam)
    -- local loss = self.encdec:validBatch(x, x_mask, y)
    local y_mask = y[{{1 + self.neu_trans.opts.decStart, -1}, {}}]:ne(0)
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
  local dataIter = BiText_Dataset.createBatch(self.src_vocab, self.dst_vocab,
    validFile, self.opts.batchSize)
  
  local totalCnt = 0
  local totalLoss = 0
  local cnt = 0
  for x, x_mask, y in dataIter do
    local loss = self.weighter:validBatch(x, x_mask, y)
    totalLoss = totalLoss + loss * x:size(2)
    totalCnt = totalCnt + y[{ {1 + self.neu_trans.opts.decStart, -1}, {} }]:ne(0):sum()
    cnt = cnt + 1
    if cnt % 5 == 0 then
      collectgarbage()
    end
  end
  
  local entropy = totalLoss / totalCnt
  local ppl =  torch.exp(entropy)
  return {entropy = entropy, ppl = ppl}
end


function Trainer:loadNeuFeatures()
  -- load encdec feature
  local encdec_opts = torch.load(self.opts.encdec_path:sub(1, -3) .. 'state.t7')
  print(encdec_opts.train)
  print(encdec_opts.valid)
  print(encdec_opts.test)
  
  self.encdec = EncDecAWE(encdec_opts)
  self.encdec:load( self.opts.encdec_path )
  xprintln('load model from %s done!', self.opts.encdec_path)
  self.weighter:addFeature('EncDecA', self.encdec)
  print '***load EncDecA done!!!***\n\n'
  self.src_vocab = encdec_opts.src_vocab
  self.dst_vocab = encdec_opts.dst_vocab
  print(table.keys( self.src_vocab ))
  print(table.keys( self.dst_vocab ))
  
  -- load neural translation feature
  local neu_trans_opts = torch.load(self.opts.neutrans_path:sub(1, -3) .. 'state.t7')
  self.neu_trans = NeuLexTransSoft(neu_trans_opts)
  self.neu_trans:load(self.opts.neutrans_path)
  xprintln('load model from %s done!', self.opts.neutrans_path)
  self.weighter:addFeature('NeuLexTransSoft', self.neu_trans)
  print '***load Neural Lexcical Translation done!!!***\n\n'
end

function Trainer:main()
  local opts = getOpts()
  print(opts)
  self.opts = opts
  
  self.weighter = NeuDynFeatWeighter(opts)
  self:loadNeuFeatures()
  
  self.trainSize, self.validSize = unpack( BiText_Dataset.getDataSize({opts.train .. '.src', opts.valid .. '.src'}) )
  printf('train size = %d, valid size = %d\n', self.trainSize, self.validSize)
  
  xprintln('Random model valid perf')
  print( self:valid(self.opts.valid) )
  
  local bestValid = {ppl = 1e309, entropy = 1e309}
  local lastValid = {ppl = 1e309, entropy = 1e309}
  local bestModel = torch.FloatTensor(self.weighter.params:size())
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
    local validRval = self:valid(self.opts.valid)
    xprint('VALID %f ', validRval.ppl)
    xprintln('lr = %.4g (%s) p = %d', opts.curLR, readableTime(timer:time().real), patience)
    
    if validRval.ppl < bestValid.ppl then
      bestValid.ppl = validRval.ppl
      bestValid.entropy = validRval.entropy
      bestValid.epoch = epoch
      self.weighter:getModel(bestModel)
    end
    
  end
  
  self.weighter:setModel(bestModel)
  opts.sgdParam = nil
  self.weighter:save(opts.save, true)
  xprintln('model saved at %s', opts.save)
  
  local validRval = self:valid(self.opts.valid)
  xprint('VALID %f \n', validRval.ppl)
  
  local testRval = self:valid(self.opts.test)
  xprint('TEST %f \n', testRval.ppl)
  
end

local function main()
  local trainer = NeuDynFeatTrainer()
  trainer:main()
end

if not package.loaded['train_neu_dyn_feat'] then
  main()
end


