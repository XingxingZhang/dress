
require 'torch'
require 'shortcut'

local LM_Dataset = torch.class('RawLM_Dataset')

LM_Dataset.SENT_END = {'###eos###'}

function LM_Dataset.createVocab(inputFile, freqCut, ignoreCase, keepFreq)
  local SENT_END = LM_Dataset.SENT_END
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
  word2idx[SENT_END[1]] = wid
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

function LM_Dataset.sent2ints(vocab, sent)
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

function LM_Dataset.getDataSize(infiles)
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

function LM_Dataset.toBatch(sents, eos, bs)
  local dtype = 'torch.LongTensor'
  local maxn = 0
  for _, sent in ipairs(sents) do
    if sent:size(1) > maxn then
      maxn = sent:size(1)
    end
  end
  maxn = maxn + 1
  local nsent = #sents
  -- for x, in default x contains EOS tokens
  local x = torch.ones(maxn, bs):type(dtype)
  -- local x = torch.ones(maxn, batchSize)
  x:mul(eos)
  local y = torch.zeros(maxn, bs):type(dtype)
  -- local y = torch.zeros(maxn, batchSize)
  for i = 1, nsent do
    local senlen = sents[i]:size(1)
    x[{ {2, senlen + 1}, i }] = sents[i]
    y[{ {1, senlen}, i }] = sents[i]
    y[{ senlen + 1, i }] = eos
  end
  
  return x, y
end

function LM_Dataset.createBatch(vocab, infile, batchSize, ncedata)
  local fin = io.open(infile)
  
  return function()
    local sents = {}
    for i = 1, batchSize do
      local line = fin:read()
      if line == nil then break end
      sents[#sents + 1] = LM_Dataset.sent2ints(vocab, line)
    end
    
    if #sents > 0 then
      local x, y = LM_Dataset.toBatch(sents, vocab.EOS, batchSize)
      if ncedata then
        -- print(x:size())
        -- print(y:size())
        local mask = y:ne(0):float()
        y[y:eq(0)] = 1
        local y_neg, y_prob, y_neg_prob = ncedata:getYNegProbs(y)
        return x, y, y_neg, y_prob, y_neg_prob, mask
      else
        return x, y
      end
    end
  end
end
