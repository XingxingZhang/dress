-- this one is used to sort
-- training data according to their length

include '../utils/shortcut.lua'
require 'paths'

local Sorter = {}

function Sorter.sortKBatches(lens, Kbatch)
  local idxs = {}
  local N = #lens
  for istart = 1, N, Kbatch do
    local iend = math.min(istart + Kbatch - 1, N)
    local subIdxs = {}
    for i = istart, iend do 
      table.insert(subIdxs, i) 
    end
    table.sort(subIdxs, function(a, b)
         return lens[b] < lens[a]  
      end)
    table.extend(idxs, subIdxs)
  end
  assert(#idxs == #lens)
  return idxs
end

function Sorter.sortBatches(lens, batchSize)
  local newIdxs = {}
  
  local len2idxs = {}
  local len2idxs_lens = {}
  for i, len in ipairs(lens) do
    local idxs = len2idxs[len]
    if idxs then
      table.insert(idxs, i)
    else
      len2idxs[len] = {i}
      table.insert(len2idxs_lens, len)
    end
  end
  
  local len2pos = {}
  for _, len in ipairs(len2idxs_lens) do
    len2pos[len] = 1
  end
  
  local pad = {}
  while true do
    local selectLen = -1
    local selectIdx = #lens + 1
    local istart, iend = -1, -1
    
    for _, len in ipairs(len2idxs_lens) do
      local pos = len2pos[len]
      local idxs = len2idxs[len]
      if pos <= #idxs and idxs[pos] < selectIdx then
        selectIdx = idxs[pos]
        selectLen = len
        istart, iend = pos, math.min(pos + batchSize - 1, #idxs)
      end
    end
    
    if selectLen == -1 then break end
    local sIdxs = len2idxs[selectLen]
    if iend - istart + 1 == batchSize then
      for i = istart, iend do
        newIdxs[#newIdxs + 1] = sIdxs[i]
      end
    else
      for i = istart, iend do
        pad[#pad + 1] = sIdxs[i]
      end
    end
    len2pos[selectLen] = iend + 1
  end -- end while
  
  table.sort(pad, function(a, b) 
      return lens[b] < lens[a]
    end)
  table.extend(newIdxs, pad)
  
  return newIdxs
end

local function getOpts()
  local cmd = torch.CmdLine()
  cmd:option('--dataset', '/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/PWKP/PWKP_108016.80', 'dataset file')
  cmd:option('--sort', -2, '0: no sorting of the training data; -1: sort training data by their length; k (k > 0): sort the consecutive k batches by their length; -2: sort all training sentence by their length')
  cmd:option('--batchSize', 64, 'batch size')
  
  local opts = cmd:parse(arg)
  assert(opts.sort == -1 or opts.sort > 0 or opts.sort == -2, 'you must sort!')
  return opts
end

local function main()
  local opts = getOpts()
  print(opts)
  
  local function getLens(infile)
    local fin = io.open(infile, 'r')
    local cnt = 0
    local lens = {}
    while true do
      local line = fin:read()
      if line == nil then break end
      cnt = cnt + 1
      local words = line:splitc(' \t\r\n')
      table.insert(lens, #words)
    end
    fin:close()
    
    return lens
  end
  
  local function writeFile(idxs, infile, outfile)
    local fin_src = io.open(infile .. '.src', 'r')
    local fin_dst = io.open(infile .. '.dst', 'r')
    local fout_src = io.open(outfile .. '.src', 'w')
    local fout_dst = io.open(outfile .. '.dst', 'w')
    
    local srcs = {}
    local dsts = {}
    while true do
      local src = fin_src:read()
      local dst = fin_dst:read()
      if src == nil then break end
      table.insert(srcs, src)
      table.insert(dsts, dst)
    end
    
    assert(#srcs == #idxs)
    for _, i in ipairs(idxs) do
      fout_src:write(srcs[i] .. '\n')
      fout_dst:write(dsts[i] .. '\n')
    end
    
    fin_src:close()
    fin_dst:close()
    fout_src:close()
    fout_dst:close()
  end
  
  -- add sort == -2 option; in this case, we sort all sentences in training set with their length
  local outLabel
  if opts.sort == -2 then
    outLabel = '.sort_2'
  else
    outLabel = (opts.sort == -1 and '.sort' or string.format('.sort%d', opts.sort))
  end
  -- for training set
  local trainDst = opts.dataset .. '.train.dst'
  local trLens = getLens(trainDst)
  local trIdxs
  if opts.sort == -2 then
    trIdxs = Sorter.sortKBatches(trLens, #trLens)
  else
    trIdxs = opts.sort == -1 and Sorter.sortBatches(trLens, opts.batchSize) or Sorter.sortKBatches(trLens, opts.sort * opts.batchSize)
  end
  local trainOutF = opts.dataset .. '.train' .. outLabel
  local trainInF = opts.dataset .. '.train'
  writeFile(trIdxs, trainInF, trainOutF)
  print 'sort training done!'
  
  local validDst = opts.dataset .. '.valid.dst'
  local vaLens = getLens(validDst)
  local vaIdxs = Sorter.sortKBatches(vaLens, #vaLens)
  local validOutF = opts.dataset .. '.valid' .. outLabel
  local validInF = opts.dataset .. '.valid'
  writeFile(vaIdxs, validInF, validOutF)
  print 'sort valid done!'
  
  local testDst = opts.dataset .. '.test.dst'
  local teLens = getLens(testDst)
  local teIdxs = Sorter.sortKBatches(teLens, #teLens)
  local testOutF = opts.dataset .. '.test' .. outLabel
  local testInF = opts.dataset .. '.test'
  writeFile(teIdxs, testInF, testOutF)
  print 'sort test done!'
end

main()

