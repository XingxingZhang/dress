
require 'torch'
require 'shortcut'

local BiText_Dataset = torch.class('BiText_Dataset')

BiText_Dataset.EOS = '###eos###'  -- for both source and target sequence

function BiText_Dataset.createVocab(inputFile, freqCut, ignoreCase, keepFreq)
  local SENT_END = BiText_Dataset.EOS
  local fin = io.open(inputFile, 'r')
  local wordVector = {}
  local wordFreq = {}
  local line_cnt = 0
  while true do
    local line = fin:read()
    if line == nil then break end
    local words = line:splitc(' \t\r\n')
    line_cnt = line_cnt + 1
    for _, word in ipairs(words) do
      if ignoreCase then word = word:lower() end
      if wordFreq[word] then
        wordFreq[word] = wordFreq[word] + 1
      else
        wordFreq[word] = 1
        wordVector[#wordVector + 1] = word
      end
    end
  end
  fin:close()
  
  local wid = 1
  local word2idx = {}
  if not wordFreq['UNK'] then
    word2idx = {UNK = wid}
    wid = wid + 1
  end
  local uniqUNK = 0
  local freqs = { 0 }
  for _, wd in ipairs(wordVector) do
    if wordFreq[wd] > freqCut then
      word2idx[wd] = wid
      freqs[wid] = wordFreq[wd]
      wid = wid + 1
    else
      uniqUNK = uniqUNK + 1
      if not wordFreq['UNK'] then
        freqs[1] = freqs[1] + wordFreq[wd]
      end
    end
  end
  word2idx[SENT_END] = wid
  freqs[wid] = line_cnt
  local vocabSize = wid
  -- wid = wid + 1
  
  local idx2word = {}
  for wd, i in pairs(word2idx) do
    idx2word[i] = wd
  end
  
  local vocab = {word2idx = word2idx, idx2word = idx2word, 
    freqCut = freqCut, ignoreCase = ignoreCase, 
    keepFreq = keepFreq, UNK = word2idx['UNK'],
    EOS = word2idx['###eos###']}
  if keepFreq then
    vocab['freqs'] = freqs
    vocab['uniqUNK'] = uniqUNK
    printf('freqs size %d\n', #freqs)
  end
  
  assert(vocabSize == table.len(word2idx))
  printf('original #words %d, after cut = %d, #words %d\n', #wordVector, freqCut, vocabSize)
  vocab['nvocab'] = vocabSize
  -- print(table.keys(vocab))
  for k, v in pairs(vocab) do
    printf('%s -- ', k)
    if type(v) ~= 'table' then
      print(v)
    else
      print('table')
    end
  end
  
  return vocab
end

function BiText_Dataset.sent2ints(vocab, sent)
  local words = sent:splitc(' \t\r\n')
  local word2idx = vocab.word2idx
  -- local eos = word2idx[LM_Dataset.SENT_END[1]]
  -- local xs = { eos }
  local xs = {}
  for _, word in ipairs(words) do
    word = vocab.ignoreCase and word:lower() or word
    local wid = word2idx[word] or vocab.UNK
    xs[#xs + 1] = wid
  end
  -- xs[#xs + 1] = eos
  
  return torch.IntTensor(xs)
end

function BiText_Dataset.getDataSize(infiles)
  local sizes = {}
  for _, infile in ipairs(infiles) do
    local fin = io.open(infile)
    local size = 0
    while true do
      local line = fin:read()
      if line == nil then break end
      size = size + 1
    end
    fin:close()
    table.insert(sizes, size)
  end
  
  return sizes
end

function BiText_Dataset.toBatch(sents_src, sents_dst, eos_src, eos_dst, bs)
  local function getMaxLen(sents)
    local maxn = 0
    for _, sent in ipairs(sents) do
      if sent:size(1) > maxn then
        maxn = sent:size(1)
      end
    end
    return maxn
  end
  
  local dtype = 'torch.LongTensor'
  
  local max_x = getMaxLen(sents_src) + 1
  local max_y = getMaxLen(sents_dst) + 2
  
  local nsent = #sents_src
  local x = torch.zeros(max_x, bs):type(dtype)
  local y = torch.zeros(max_y, bs):type(dtype)
  
  local function reverse(a)
    local b = torch.Tensor(a:size()):typeAs(a)
    local N = a:size(1)
    for i = 1, N do
      b[i] = a[N+1-i]
    end
    return b
  end
  
  for i = 1, nsent do
    local senlen_src = sents_src[i]:size(1)
    local senlen_dst = sents_dst[i]:size(1)
    --[[
    x[{ max_x - senlen_src, i }] = eos_src
    x[{ {max_x - senlen_src + 1, max_x}, i }] = reverse( sents_src[i] )
    --]]
    x[{ 1, i }] = eos_src
    x[{ {2, senlen_src + 1}, i }] = reverse( sents_src[i] )
    
    y[{1, i }] = eos_dst
    y[{ {2, senlen_dst + 1}, i }] = sents_dst[i]
    y[{ senlen_dst + 2, i }] = eos_dst
  end
  
  return x, y
end

function BiText_Dataset.createBatch(vocab_src, vocab_dst, infile, batchSize, ncedata)
  local fin_src = io.open(infile .. '.src')
  local fin_dst = io.open(infile .. '.dst')
  
  return function()
    local sents_src = {}
    local sents_dst = {}
    for i = 1, batchSize do
      local line_src = fin_src:read()
      local line_dst = fin_dst:read()
      if line_src == nil then break end
      sents_src[#sents_src + 1] = BiText_Dataset.sent2ints(vocab_src, line_src)
      sents_dst[#sents_dst + 1] = BiText_Dataset.sent2ints(vocab_dst, line_dst)
    end
    
    if #sents_src > 0 then
      local x, y = BiText_Dataset.toBatch(sents_src, sents_dst, vocab_src.EOS, vocab_dst.EOS, batchSize)
      local x_mask = x:ne(0):float()
      x[x:eq(0)] = 1
      
      if ncedata then
        local y_mask = y:ne(0):float()
        y[y:eq(0)] = 1
        local y_neg, y_prob, y_neg_prob = ncedata:getYNegProbs(y)
        return x, x_mask, y, y_mask, y_neg, y_prob, y_neg_prob
      else
        return x, x_mask, y
      end
    else
      fin_src:close()
      fin_dst:close()
    end
  end
end

function BiText_Dataset.createBatchShuffle(vocab_src, vocab_dst, infile, batchSize, ncedata)
  local fin_src = io.open(infile .. '.src')
  local fin_dst = io.open(infile .. '.dst')
  
  local all_sents_src = {}
  local all_sents_dst = {}
  local all_sent_count = 0
  while true do
    local line_src = fin_src:read()
    local line_dst = fin_dst:read()
    if line_src == nil then
      assert(line_dst == nil) 
      break
    end
    
    table.insert( all_sents_src, BiText_Dataset.sent2ints(vocab_src, line_src) )
    table.insert( all_sents_dst, BiText_Dataset.sent2ints(vocab_dst, line_dst) )
    all_sent_count = all_sent_count + 1
  end
  fin_src:close()
  fin_dst:close()
  
  local tmp_all_sents = {}
  local rIdxs = torch.randperm(all_sent_count)
  for i = 1, all_sent_count do
    local idx = rIdxs[i]
    table.insert(tmp_all_sents, {all_sents_src[idx], all_sents_dst[idx]})
  end
  for i = 1, all_sent_count do
    local xy = tmp_all_sents[i]
    all_sents_src[i] = xy[1]
    all_sents_dst[i] = xy[2]
  end
  
  local nbatches = math.ceil( #all_sents_src / batchSize )
  local rndIdxs = torch.randperm(nbatches)
  local offset = (torch.random() % batchSize) + 1
  local count = 0
  
  return function()
    
    count = count + 1
    if count <= nbatches then
      -- load a batch of sentences
      local sents_src = {}
      local sents_dst = {}
      local rndIdx = (rndIdxs[count]-1) * batchSize + offset
      for j = rndIdx, rndIdx + batchSize - 1 do
        local idx = (j-1)%all_sent_count + 1
        sents_src[#sents_src + 1] = all_sents_src[idx] -- BiText_Dataset.sent2ints(vocab_src, all_sents_src[idx])
        sents_dst[#sents_dst + 1] = all_sents_dst[idx] -- BiText_Dataset.sent2ints(vocab_dst, all_sents_dst[idx])
      end
      -- create batches
      local x, y = BiText_Dataset.toBatch(sents_src, sents_dst, vocab_src.EOS, vocab_dst.EOS, batchSize)
      local x_mask = x:ne(0):float()
      x[x:eq(0)] = 1
      
      if ncedata then
        local y_mask = y:ne(0):float()
        y[y:eq(0)] = 1
        local y_neg, y_prob, y_neg_prob = ncedata:getYNegProbs(y)
        return x, x_mask, y, y_mask, y_neg, y_prob, y_neg_prob
      else
        return x, x_mask, y
      end
    end
    
  end
end

function BiText_Dataset.toBatchX(sents_src, eos_src)
  local function getMaxLen(sents)
    local maxn = 0
    for _, sent in ipairs(sents) do
      if sent:size(1) > maxn then
        maxn = sent:size(1)
      end
    end
    return maxn
  end
  
  local dtype = 'torch.LongTensor'
  
  local max_x = getMaxLen(sents_src) + 1
  
  local nsent = #sents_src
  local bs = nsent
  local x = torch.zeros(max_x, bs):type(dtype)
  
  local function reverse(a)
    local b = torch.Tensor(a:size()):typeAs(a)
    local N = a:size(1)
    for i = 1, N do
      b[i] = a[N+1-i]
    end
    return b
  end
  
  for i = 1, nsent do
    local senlen_src = sents_src[i]:size(1)
    --[[
    x[{ max_x - senlen_src, i }] = eos_src
    x[{ {max_x - senlen_src + 1, max_x}, i }] = reverse( sents_src[i] )
    --]]
    x[{ 1, i }] = eos_src
    x[{ {2, senlen_src + 1}, i }] = reverse( sents_src[i] )
  end
  
  return x
end

function BiText_Dataset.createBatchX(vocab_src, infile, batchSize)
  local fin_src = io.open(infile .. '.src')
  
  return function()
    local sents_src = {}
    for i = 1, batchSize do
      local line_src = fin_src:read()
      if line_src == nil then break end
      sents_src[#sents_src + 1] = BiText_Dataset.sent2ints(vocab_src, line_src)
    end
    
    if #sents_src > 0 then
      local x = BiText_Dataset.toBatchX(sents_src, vocab_src.EOS)
      local x_mask = x:ne(0):float()
      x[x:eq(0)] = 1
      
      return x, x_mask
    else
      fin_src:close()
    end
  end
end
