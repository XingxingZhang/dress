

local Lex_Dataset = torch.class('LexTrans_Dataset')

function Lex_Dataset.createDataset(src_vocab, dst_vocab, infile)
  local raw_data = torch.load(infile)
  xprintln('load data done!')
  
  local function towid(vocab, wd)
    return vocab.word2idx[wd] or vocab.UNK
  end
  
  local function tostd(split)
    local std_split = {}
    for i, sent in ipairs(split) do
      local src = {src_vocab.EOS}
      local dst = {dst_vocab.EOS}
      for j = #sent.src, 1, -1 do
        table.insert(src, towid(src_vocab, sent.src[j]))
      end
      for j = 1, #sent.dst do
        table.insert(dst, towid(dst_vocab, sent.dst[j]))
      end
      table.insert(dst, dst_vocab.EOS)
      local std_sent = {src = torch.LongTensor(src), dst = torch.LongTensor(dst), 
        align = sent.align, detailed_align = sent.detailed_align}
      assert(std_sent.align:size(1) == std_sent.dst:size(1) - 1)
      std_split[#std_split+1] = std_sent
    end
    
    return std_split
  end
  
  return {train = tostd(raw_data.train), valid = tostd(raw_data.valid), test = tostd(raw_data.test)}
end

function Lex_Dataset.shuffle(data_split)
  local N = #data_split
  local tmp_split = {}
  local idxs = torch.randperm(N)
  for i = 1, N do
    local idx = idxs[i]
    tmp_split[#tmp_split + 1] = data_split[idx]
  end
  return tmp_split
end

function Lex_Dataset.toBatch(batch, batchSize)
  local xmax, ymax, amax = 0, 0, 0
  for _, sent in ipairs(batch) do
    xmax = math.max(sent.src:size(1), xmax)
    ymax = math.max(sent.dst:size(1), ymax)
    amax = math.max(sent.align:size(1), amax)
  end
  local x = torch.LongTensor(xmax, batchSize):fill(0)
  local a = torch.LongTensor(amax, batchSize):fill(0)
  local y = torch.LongTensor(ymax, batchSize):fill(0)
  for i, sent in ipairs(batch) do
    local src, dst, align = sent.src, sent.dst, sent.align
    x[{ {1, src:size(1)}, i }] = src
    a[{ {1, align:size(1)}, i }] = align
    y[{ {1, dst:size(1)}, i }] = dst
  end
  return x, a, y
end

function Lex_Dataset.createBatch(data_split, batchSize, UNK, DEF_POS, shuffle)
  local data = shuffle and Lex_Dataset.shuffle(data_split) or data_split
  local N, istart = #data, 1
  
  return function()
    if istart <= N then
      local iend = math.min(N, istart + batchSize - 1)
      local batch = {}
      for i = istart, iend do
        batch[#batch + 1] = data[i]
      end
      istart = iend + 1
      
      local x, a, y = Lex_Dataset.toBatch(batch, batchSize)
      local x_mask = x:ne(0):float()
      x[x:eq(0)] = UNK
      a[a:eq(0)] = DEF_POS
      
      return x, x_mask, a, y
    end
  end
end


function Lex_Dataset.toBatchStochastic(batch, batchSize)
  local xmax, ymax, amax = 0, 0, 0
  for _, sent in ipairs(batch) do
    xmax = math.max(sent.src:size(1), xmax)
    ymax = math.max(sent.dst:size(1), ymax)
    amax = math.max(sent.align:size(1), amax)
  end
  local x = torch.LongTensor(xmax, batchSize):fill(0)
  local a = torch.LongTensor(amax, batchSize):fill(0)
  local y = torch.LongTensor(ymax, batchSize):fill(0)
  for i, sent in ipairs(batch) do
    local dalign = sent.detailed_align
    local align = torch.multinomial(dalign, 1, true):view(-1)
    local src, dst = sent.src, sent.dst
    x[{ {1, src:size(1)}, i }] = src
    a[{ {1, align:size(1)}, i }] = align
    y[{ {1, dst:size(1)}, i }] = dst
  end
  return x, a, y
end


function Lex_Dataset.createBatchStochastic(data_split, batchSize, UNK, DEF_POS, shuffle)
  local data = shuffle and Lex_Dataset.shuffle(data_split) or data_split
  local N, istart = #data, 1
  
  return function()
    if istart <= N then
      local iend = math.min(N, istart + batchSize - 1)
      local batch = {}
      for i = istart, iend do
        batch[#batch + 1] = data[i]
      end
      istart = iend + 1
      
      local x, a, y = Lex_Dataset.toBatchStochastic(batch, batchSize)
      local x_mask = x:ne(0):float()
      x[x:eq(0)] = UNK
      a[a:eq(0)] = DEF_POS
      
      return x, x_mask, a, y
    end
  end
end


function Lex_Dataset.toBatchSoft(batch, batchSize)
  local xmax, ymax, amax = 0, 0, 0
  for _, sent in ipairs(batch) do
    xmax = math.max(sent.src:size(1), xmax)
    ymax = math.max(sent.dst:size(1), ymax)
    amax = math.max(sent.align:size(1), amax)
  end
  local x = torch.LongTensor(xmax, batchSize):fill(0)
  local a = torch.FloatTensor(amax, batchSize, xmax):fill(0)
  local y = torch.LongTensor(ymax, batchSize):fill(0)
  for i, sent in ipairs(batch) do
    local dalign = sent.detailed_align
    -- local align = torch.multinomial(dalign, 1, true):view(-1)
    local src, dst = sent.src, sent.dst
    x[{ {1, src:size(1)}, i }] = src
    a[{ {1, dalign:size(1)}, i, {1, dalign:size(2)} }] = dalign
    y[{ {1, dst:size(1)}, i }] = dst
  end
  return x, a, y
end


function Lex_Dataset.createBatchSoft(data_split, batchSize, UNK, DEF_POS, shuffle)
  local data = shuffle and Lex_Dataset.shuffle(data_split) or data_split
  local N, istart = #data, 1
  
  return function()
    if istart <= N then
      local iend = math.min(N, istart + batchSize - 1)
      local batch = {}
      for i = istart, iend do
        batch[#batch + 1] = data[i]
      end
      istart = iend + 1
      
      local x, a, y = Lex_Dataset.toBatchSoft(batch, batchSize)
      local x_mask = x:ne(0):float()
      x[x:eq(0)] = UNK
      -- a[a:eq(0)] = DEF_POS
      
      return x, x_mask, a, y
    end
  end
end


