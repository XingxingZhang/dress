
require '.'
require 'shortcut'
require 'paths'

require 'NCEDataGenerator'
require 'BiText_Dataset'
require 'EncDec'
require 'EncDecA'
require 'EncDecAWE'
require 'EncDecARF'
require 'EncDecARF_Plus'

local Trainer = torch.class('EncDecTrainer')

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
  local dataIter = BiText_Dataset.createBatchShuffle(self.opts.src_vocab, self.opts.dst_vocab, 
    self.opts.train, self.opts.batchSize)
  local dataSize, curDataSize = self.trainSize, 0
  local percent, inc = 0.001, 0.001
  local timer = torch.Timer()
  -- local sgdParam = {learningRate = opts.curLR}
  local sgdParam = self.opts.sgdParam
  local cnt = 0
  local totalLoss = 0
  local totalCnt = 0
  local totalLossRF, totalCntRF = 0, 0
  for x, x_mask, y in dataIter do
    local train_info = self.encdec:trainBatch(x, x_mask, y, sgdParam)
    local loss = train_info.nll_loss
    
    if self.samplePos > 1 then
      local end_pos = math.min(self.samplePos, y:size(1))
      local y_mask = y[{{2, end_pos}, {}}]:ne(0)
      local nll = loss * x:size(2) / (y_mask:sum())
      
      totalLoss = totalLoss + loss * x:size(2)
      totalCnt = totalCnt + y_mask:sum()
    end
    
    totalLossRF = totalLossRF + train_info.rf_loss * x:size(2)
    totalCntRF = totalCntRF + train_info.n_seq
    
    curDataSize = curDataSize + x:size(2)
    local ratio = curDataSize/dataSize
    if ratio >= percent then
      local wps = totalCnt / timer:time().real
      xprint( '\r%s %.3f nll: %.4f rf: %.4f (%s) / %.2f wps ... ', EPOCH_INFO, ratio, totalLoss/totalCnt, 
        totalLossRF/totalCntRF, readableTime(timer:time().real), wps )
      
      percent = math.floor(ratio / inc) * inc
      percent = percent + inc
    end
    
    cnt = cnt + 1
    if cnt % 5 == 0 then
      collectgarbage()
    end
    
    -- this is for test
    -- if cnt == 20 then break end
  end
  
  local r1 = totalCnt == 0 and 0 or totalLoss / totalCnt
  return r1, totalLossRF/totalCntRF
end

function Trainer:valid(validFile)
  -- self.encdec:disableDropout()
  --[[
  local dataIter = BiText_Dataset.createBatch(self.opts.src_vocab, self.opts.dst_vocab, 
    self.opts.train, self.opts.batchSize, self.ncedata)
  --]]
  local dataIter = BiText_Dataset.createBatch(self.opts.src_vocab, self.opts.dst_vocab,
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

function Trainer:validBLEU(validFile, tmpFile)
  -- self.encdec:disableDropout()
  local dataIter = BiText_Dataset.createBatchX(self.opts.src_vocab, validFile, self.opts.validBatchSize)
  local fout = io.open(tmpFile, 'w')
  for x, x_mask in dataIter do
    local sents = self.encdec:sample(x, x_mask)
    for _, sent in ipairs(sents) do
      fout:write(sent .. '\n')
    end
  end
  -- self.encdec:enableDropout()
  fout:close()
  
  local cmd = string.format('./scripts/multi-bleu.perl %s < %s', validFile .. '.dst', tmpFile)
  os.execute(cmd)
end


function Trainer:validSample(validFile, greedy)
  -- validBatchSample
  local dataIter = BiText_Dataset.createBatch(self.opts.src_vocab, self.opts.dst_vocab,
    validFile, self.opts.validBatchSize)
  local totalCnt = 0
  local totalLoss = 0
  local cnt = 0
  local sariLoss, lmLoss, simLoss = 0, 0, 0
  for x, x_mask, y in dataIter do
    local reward, size, r_sari, r_lm, r_sim  = self.encdec:validBatchSample(x, x_mask, y, greedy)
    totalLoss = totalLoss + reward
    totalCnt = totalCnt + size
    
    sariLoss = sariLoss + r_sari
    lmLoss = lmLoss + r_lm
    simLoss = simLoss + r_sim
    
    cnt = cnt + 1
    if cnt % 5 == 0 then
      collectgarbage()
    end
  end
  
  return totalLoss / totalCnt, sariLoss / totalCnt, lmLoss / totalCnt, simLoss / totalCnt
end


function Trainer:main()
  local model_opts = require 'model_opts'
  local opts = model_opts.getOpts()
  self.opts = opts
  
  local srcVocabPath = opts.train .. '.src.vocab.tmp.t7'
  local dstVocabPath = opts.train .. '.dst.vocab.tmp.t7'
  if paths.filep(srcVocabPath) then
    opts.src_vocab = torch.load(srcVocabPath)
    opts.dst_vocab = torch.load(dstVocabPath)
    printf('src vocab size = %d, dst vocab size = %d\n', opts.src_vocab.nvocab, opts.dst_vocab.nvocab)
  else
    opts.src_vocab = BiText_Dataset.createVocab(opts.train .. '.src', opts.freqCut, opts.ignoreCase, true)
    opts.dst_vocab = BiText_Dataset.createVocab(opts.train .. '.dst', opts.freqCut, opts.ignoreCase, true)
    print 'creat vocab done!'
    torch.save(srcVocabPath, opts.src_vocab)
    torch.save(dstVocabPath, opts.dst_vocab)
    print 'save vocab done!'
  end
  printf('src vocab EOS ID is %d, sym is %s; dst vocab EOS ID is %d, sym is %s\n', opts.src_vocab.EOS,
      opts.src_vocab.idx2word[opts.src_vocab.EOS], opts.dst_vocab.EOS, opts.dst_vocab.idx2word[opts.dst_vocab.EOS])
  
  self.trainSize, self.validSize = unpack( BiText_Dataset.getDataSize({opts.train .. '.src', opts.valid .. '.src'}) )
  printf('train size = %d, valid size = %d\n', self.trainSize, self.validSize)
  
  -- self.ncedata = NCEDataGenerator(opts.dst_vocab, opts.nneg, opts.power, opts.normalizeUNK)
  if opts.model == 'EncDec' then
    self.encdec = EncDec(opts)
  elseif opts.model == 'EncDecA' then
    self.encdec = EncDecA(opts)
  elseif opts.model == 'EncDecAWE' then
    self.encdec = EncDecAWE(opts)
  elseif opts.model == 'EncDecALN' then
    self.encdec = EncDecALN(opts)
  elseif opts.model == 'EncDecARF' then
    self.encdec = EncDecARF(opts)
  elseif opts.model == 'EncDecARF_Plus' then
    self.encdec = EncDecARF_Plus(opts)
  end
  self:showOpts()
  self.encdec:loadSub(opts.encdecPath)
  local encdec = self.encdec
  self.samplePos = opts.sampleStart
  
  
  if opts.model == 'EncDecARF_Plus' then
    self.encdec:loadLMScorer(opts.lmPath)
    self.encdec:loadSimScorer(opts.simPath)
  end
  
  
  local bestValid = {ppl = 1e309, entropy = 1e309}
  local lastValid = {ppl = 1e309, entropy = 1e309}
  local bestModel = torch.FloatTensor(self.encdec.params:size())
  local patience = opts.patience
  local divLR = false
  local timer = torch.Timer()
  local epochNo = 0
  for epoch = 1, opts.maxEpoch do
    if self.samplePos < 1 then
      xprintln('curriculum learning done! samplePos = %d', self.samplePos)
      break
    end
    
    encdec:setSampleStart(self.samplePos)
    
    epochNo = epochNo + 1
    EPOCH_INFO = string.format('epoch %d rf_start %d ', epoch, self.samplePos)
    
    local startTime = timer:time().real
    local trainCost, trainCostRF = self:train()
    xprint('\repoch %d TRAIN nll %f rf %f ', epoch, trainCost, trainCostRF)
    local validRval = self:valid(opts.valid)
    xprint('VALID %f - ', validRval.ppl)
    local validReward, valid_sari, valid_lm, valid_sim = self:validSample(opts.valid)
    xprint('%f %f (sar) %f (lm) %f (sim) | ', validReward, valid_sari, valid_lm, valid_sim)
    local validReward_greedy, validR_sari, validR_lm, validR_sim = self:validSample(opts.valid, true)
    xprint('%f (grd) %f (sar) %f (lm) %f (sim) ', validReward_greedy, validR_sari, validR_lm, validR_sim)
    local endTime = timer:time().real
    xprintln('lr = %.4g (%s) rf_start = %d', opts.curLR, readableTime(endTime - startTime), self.samplePos)
    
    if not paths.dirp(opts.save) then
      paths.mkdir(opts.save)
    end
    local save_path = string.format('%s/%d.t7', opts.save, epochNo)
    encdec:save(save_path, true)
    
    if epoch % opts.rfEpoch == 0 then
      self.samplePos = self.samplePos - opts.deltaSamplePos
    end
    
  end
  
  if epochNo >= opts.maxEpoch then
    xprintln('Max number of epoch is met. Training finished!')
  end
  
end

local function main()
  local trainer = EncDecTrainer()
  trainer:main()
end

main()

