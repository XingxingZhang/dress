
require '.'
require 'shortcut'
require 'hdf5'
require 'LSTMNCELM'
require 'RawLM_Dataset'

local model_utils = require 'model_utils'

-- currently only support LSTMNCELM
local Reranker = torch.class('LSTMLMReranker')

local function showOpts(opts)
  local tmp = opts.vocab
  opts.vocab = nil
  print(opts)
  opts.vocab = tmp
end

function Reranker:__init(modelPath, useGPU)
  if useGPU then
    require 'cutorch'
    require 'cunn'
  end
  
  local optsPath = modelPath:sub(1, -4) .. '.state.t7'
  print(optsPath)
  local opts = torch.load(optsPath)
  self.opts = opts
  xprintln('load state from %s done!', optsPath)
  
  opts.useGPU = useGPU
  showOpts(opts)
  
  torch.manualSeed(opts.seed)
  if opts.useGPU then
    cutorch.manualSeed(opts.seed)
  end
  
  if opts.model == 'LSTMNCE' then
    self.rnnlm = LSTMNCELM(opts)
    -- error('currently only support LSTMNCELM')
  end
  
  xprintln( 'load model from %s', modelPath )
  self.rnnlm:load(modelPath)
  xprintln( 'load model from %s done!', modelPath )
end

function Reranker:rerank(testFile, outFile, batchSize)
  self.rnnlm:disableDropout()
  
  local logp_sents = {}
  local dataIter = RawLM_Dataset.createBatch(self.opts.vocab, testFile, batchSize)
  
  local cnt = 0
  for x, y in dataIter do
    -- yPred | size: (seqlen*bs, nvocab)
    -- local _tmp, yPred
    local  _tmp, yPred = self.rnnlm:validBatch(x, y)
    if self.opts.useGPU then y = y:cuda() end
    -- local mask = y:ne(0):double()
    local mask = self.opts.useGPU and y:ne(0):cuda() or y:ne(0):double()
    y[y:eq(0)] = 1
    local y_ = y:view(y:size(1) * y:size(2), 1)
    local logps = yPred:gather(2, y_)   -- shape: seqlen*bs, 1
    local logp_sents_ = logps:cmul(mask):view(y:size(1), y:size(2)):sum(1):squeeze()
    for i = 1, logp_sents_:size(1) do
      logp_sents[#logp_sents + 1] = logp_sents_[i]
    end
    
    cnt = cnt + y:size(2)
    if cnt % 100 == 0 then
      xprintln('cnt = %d', cnt)
    end
  end
  
  local lmfile = './msr_scripts/Holmes.lm_format.questions.txt'
  local lines = xreadlines(lmfile)
  assert(#lines == #logp_sents, 'there should be the same number of sentences in testFile and lmfile')
  
  local fout = io.open(outFile, 'w')
  for i = 1, #lines do
    fout:write( string.format('%s\t%f\n', lines[i]:trim(), logp_sents[i]) )
  end
  fout:close()
  
  local accFile = outFile .. '.acc'
  Reranker.score2accuracy(outFile, accFile)
  
  self.rnnlm:enableDropout()
end

function Reranker.score2accuracy(scoreFile, accFile)
  local bestof5_pl = './msr_scripts/bestof5.pl'
  local score_pl = './msr_scripts/score.pl'
  local ans_file = './msr_scripts/Holmes.lm_format.answers.txt'
  local tmp_file = scoreFile .. '.__sample.temp__'
  local cmd = string.format('cat %s | %s > %s', scoreFile, bestof5_pl, tmp_file)
  os.execute(cmd)
  cmd = string.format('%s %s %s > %s', score_pl, tmp_file, ans_file, accFile)
  os.execute(cmd)
  
  local lines = xreadlines(accFile)
  local nLines = #lines
  for i = nLines - 4, nLines do
    print(lines[i])
  end
end

--[[
function Reranker.toBatch(xs, ys, batchSize)
  local dtype = 'torch.LongTensor'
  local maxn = 0
  for _, y_ in ipairs(ys) do
    if y_:size(1) > maxn then
      maxn = y_:size(1)
    end
  end
  local x = torch.ones(maxn, batchSize, 4):type(dtype)
  -- x:mul(1)
  x[{ {}, {}, 4 }] = torch.linspace(2, maxn + 1, maxn):resize(maxn, 1):expand(maxn, batchSize)
  local nsent = #ys
  local y = torch.zeros(maxn, batchSize):type(dtype)
  for i = 1, nsent do
    local sx, sy = xs[i], ys[i]
    x[{ {1, sx:size(1)}, i, {} }] = sx
    y[{ {1, sy:size(1)}, i }] = sy
  end
  
  return x, y
end

function Reranker.createBatch(testH5File, batchSize)
  local h5in = hdf5.open(testH5File, 'r')
  local label = 'test'
  local x_data = h5in:read(string.format('/%s/x_data', label))
  local y_data = h5in:read(string.format('/%s/y_data', label))
  local index = h5in:read(string.format('/%s/index', label))
  local N = index:dataspaceSize()[1]
  
  local istart = 1
  
  return function()
    if istart <= N then
      local iend = math.min(istart + batchSize - 1, N)
      local xs = {}
      local ys = {}
      for i = istart, iend do
        local idx = index:partial({i, i}, {1, 2})
        local start, len = idx[1][1], idx[1][2]
        local x = x_data:partial({start, start + len - 1}, {1, 4})
        local y = y_data:partial({start, start + len - 1})
        table.insert(xs, x)
        table.insert(ys, y)
      end
      
      istart = iend + 1
      
      local x, y = Reranker.toBatch(xs, ys, batchSize)
      
      return x, y
    else
      h5in:close()
    end
  end
end

function Reranker.toBatchBidirectional(xs, ys, lcs, batchSize)
  local dtype = 'torch.LongTensor'
  local maxn = 0
  for _, y_ in ipairs(ys) do
    if y_:size(1) > maxn then
      maxn = y_:size(1)
    end
  end
  local x = torch.ones(maxn, batchSize, 5):type(dtype)
  -- x:mul(self.UNK)
  x[{ {}, {}, 4 }] = torch.linspace(2, maxn + 1, maxn):resize(maxn, 1):expand(maxn, batchSize)
  x[{ {}, {}, 5 }] = 0    -- in default, I don't want them to have 
  local nsent = #ys
  local y = torch.zeros(maxn, batchSize):type(dtype)
  for i = 1, nsent do
    local sx, sy = xs[i], ys[i]
    x[{ {1, sx:size(1)}, i, {} }] = sx
    y[{ {1, sy:size(1)}, i }] = sy
  end
  
  -- for left children
  assert(#lcs == #xs, 'should be the same!')
  local lcBatchSize = 0
  local maxLcSeqLen = 0
  for _, lc in ipairs(lcs) do
    if lc:dim() ~= 0 then
      lcBatchSize = lcBatchSize + 1
      maxLcSeqLen = math.max(maxLcSeqLen, lc:size(1))
    end
  end
  local lchild = torch.Tensor():type(dtype)
  local lc_mask = torch.FloatTensor()
  
  if lcBatchSize ~= 0 then
    lchild:resize(maxLcSeqLen, lcBatchSize):fill(1)   -- UNK should be 1
    lc_mask:resize(maxLcSeqLen, lcBatchSize):fill(0)
    local j = 0
    for i, lc in ipairs(lcs) do
      if lc:dim() ~= 0 then
        j = j + 1
        lchild[{ {1, lc:size(1)}, j }] = lc[{ {}, 1 }]
        lc_mask[{ {1, lc:size(1)}, j }] = lc[{ {}, 2 }] + 1
        local xcol = x[{ {}, i, 5 }]
        local idxs = xcol:ne(0)
        xcol[idxs] = (xcol[idxs] - 1) * lcBatchSize + j
      end
    end
  end
  
  return x, y, lchild, lc_mask
end

function Reranker.createBatchBidirectional(testH5File, batchSize)
  local h5in = hdf5.open(testH5File, 'r')
  local label = 'test'
  local x_data = h5in:read(string.format('/%s/x_data', label))
  local y_data = h5in:read(string.format('/%s/y_data', label))
  local index = h5in:read(string.format('/%s/index', label))
  local l_data = h5in:read( string.format('/%s/l_data', label) )
  local lindex = h5in:read( string.format('/%s/lindex', label) )
  local N = index:dataspaceSize()[1]
  
  local istart = 1
  
  return function()
    if istart <= N then
      local iend = math.min(istart + batchSize - 1, N)
      local xs = {}
      local ys = {}
      local lcs = {}
      for i = istart, iend do
        local idx = index:partial({i, i}, {1, 2})
        local start, len = idx[1][1], idx[1][2]
        local x = x_data:partial({start, start + len - 1}, {1, 5})
        local y = y_data:partial({start, start + len - 1})
        table.insert(xs, x)
        table.insert(ys, y)
        
        local lidx = lindex:partial({i, i}, {1, 2})
        local lstart, llen = lidx[1][1], lidx[1][2]
        local lc
        if llen == 0 then
          lc = torch.IntTensor()  -- to be the same type as l_data
        else
          lc = l_data:partial({lstart, lstart + llen - 1}, {1, 2})
        end
        table.insert(lcs, lc)
      end
      
      istart = iend + 1
      
      return Reranker.toBatchBidirectional(xs, ys, lcs, batchSize)
    else
      h5in:close()
    end
  end
end
--]]

local function getOpts()
  local cmd = torch.CmdLine()
  cmd:text('====== Reranking for MSR sentence completion challenge ======')
  cmd:option('--useGPU', false, 'do you want to run this on a GPU?')
  cmd:option('--modelPath', '', 'path for the trained model; modelPath.state.t7 should be the option of the model')
  cmd:option('--testFile', '', 'test file for reranking (.h5)')
  cmd:option('--outFile', '', 'path for the output file')
  cmd:option('--batchSize', 50, 'batch size')
  
  return cmd:parse(arg)
end

local function main()
  local opts = getOpts()
  assert(5200 % opts.batchSize == 0)
  print(opts)
  local reranker = LSTMLMReranker(opts.modelPath, opts.useGPU)
  reranker:rerank(opts.testFile, opts.outFile, opts.batchSize)
end

main()

