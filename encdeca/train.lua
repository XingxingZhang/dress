
require '.'
require 'shortcut'
require 'paths'

require 'NCEDataGenerator'
require 'BiText_Dataset'
require 'EncDec'
require 'EncDecA'
require 'EncDecAWE'

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
  --[[
  local dataIter = BiText_Dataset.createBatch(self.opts.src_vocab, self.opts.dst_vocab, 
    self.opts.train, self.opts.batchSize)
  --]]
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
  
  self.ncedata = NCEDataGenerator(opts.dst_vocab, opts.nneg, opts.power, opts.normalizeUNK)
  if opts.model == 'EncDec' then
    self.encdec = EncDec(opts)
  elseif opts.model == 'EncDecA' then
    self.encdec = EncDecA(opts)
  elseif opts.model == 'EncDecAWE' then
    self.encdec = EncDecAWE(opts)
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
  
  if epochNo > opts.maxEpoch then
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
  local trainer = EncDecTrainer()
  trainer:main()
end

main()



