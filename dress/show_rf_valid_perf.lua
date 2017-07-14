
require '.'
require 'shortcut'
require 'paths'

require 'BiText_Dataset'
-- require 'EncDecARF'
require 'EncDecARF_Plus'

local Trainer = torch.class('EncDecTrainer')


function Trainer:validSample(validFile)
  -- validBatchSample
  local dataIter = BiText_Dataset.createBatch(self.opts.src_vocab, self.opts.dst_vocab,
    validFile, self.opts.validBatchSize)
  local totalCnt = 0
  local totalLoss = 0
  local cnt = 0
  local sariLoss, lmLoss, simLoss = 0, 0, 0
  
  for x, x_mask, y in dataIter do
    local reward, size, r_sari, r_lm, r_sim  = self.encdec:validBatchSample(x, x_mask, y, self.opts.greedyInference)
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
  local cmd = torch.CmdLine()
  cmd:option('--dir', '/disk/scratch1/xingxing.zhang/seq2seq/sent_simple/encdec_rl/test/model_0.01.lr0.01.rf2', '')
  cmd:option('--greedy', false, '')
  local args = cmd:parse(arg)
  
  local opts = torch.load(args.dir .. '/1.state.t7')
  self.opts = opts
  require 'cutorch'
  require 'cunn'
  torch.manualSeed(opts.seed)
  cutorch.manualSeed(opts.seed)
  
  opts.greedyInference = args.greedy
  
  self.encdec = EncDecARF_Plus(opts)
  if opts.model == 'EncDecARF_Plus' then
    self.encdec:loadLMScorer(opts.lmPath)
    self.encdec:loadSimScorer(opts.simPath)
  end
  
  local epoch = 1
  local validPerf = {}
  local testPerf = {}
  while true do
    local model_path = string.format('%s/%d.t7', args.dir, epoch)
    if not paths.filep(model_path) then
      break
    end
    self.encdec:load(model_path)
    local validReward, validSari, validLM, validSim
    local testReward, testSari, testLM, testSim
    local n_test = 1
    for i = 1, n_test do
      validReward, validSari, validLM, validSim = self:validSample(opts.valid)
      testReward, testSari, testLM, testSim = self:validSample(opts.test)
      xprintln('epoch %d valid reward = %f sari = %f, lm = %f, sim = %f | test reward = %f sari = %f lm = %f sim = %f', epoch, 
        validReward, validSari, validLM, validSim,
        testReward, testSari, testLM, testSim)
    end
    
    validPerf[#validPerf + 1] = validReward
    testPerf[#testPerf + 1] = testReward
    
    epoch = epoch + 1
  end
  
  local tmp = xmap(tostring, validPerf)
  print( table.concat(tmp, ', ') )
  tmp = xmap(tostring, testPerf)
  print( table.concat(tmp, ', ') )
  
end

local function main()
  local trainer = EncDecTrainer()
  trainer:main()
end

main()

