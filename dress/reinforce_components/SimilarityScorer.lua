
require 'shortcut'
require 'EncDec'
require 'UniText_Dataset'

local Scorer = torch.class('SimilarityScorer')


function Scorer:__init(auto_enc_path, batch_size)
  local opts = torch.load(auto_enc_path:sub(1, -3) .. 'state.t7')
  self.opts = opts
  torch.manualSeed(self.opts.seed)
  if self.opts.useGPU then
    require 'cutorch'
    require 'cunn'
    cutorch.manualSeed(self.opts.seed)
  end
  if batch_size then
    self.opts.batchSize = batch_size
  end
  
  self:showOpts(self.opts)
  self.auto_encoder = EncDec(self.opts)
  self.auto_encoder:load(auto_enc_path)
  
  print 'create and load Sequence Auto Encoder done!'
end


function Scorer:showOpts(opts)
  for k, v in pairs(opts) do
    if torch.type(v) == 'table' then
      xprintln('%s -- table', k)
    else
      xprintln('%s -- %s', k, tostring(v))
    end
  end
end


function Scorer:toBatch(sents_src, eos_src, bs, UNK)
  local function getMaxLen(sents)
    local maxn = 0
    for _, sent in ipairs(sents) do
      if sent:dim() == 1 and sent:size(1) > maxn then
        maxn = sent:size(1)
      end
    end
    return maxn
  end
  
  local dtype = 'torch.LongTensor'
  
  local max_x = getMaxLen(sents_src) + 1
  
  local nsent = #sents_src
  local x = torch.zeros(max_x, bs):type(dtype)
  
  for i = 1, nsent do
    if sents_src[i]:dim() ~= 0 then
      local senlen_src = sents_src[i]:size(1)
      x[{ 1, i }] = eos_src
      x[{ {2, senlen_src + 1}, i }] = sents_src[i]
    end
  end
  
  local x_mask = x:ne(0):float()
  x[x:eq(0)] = UNK
  
  return x, x_mask
end


function Scorer:score(sentsA, sentsB, reprA, reprB)
  assert(#sentsA == #sentsB, 'must have the same number of sentences!')
  
  reprA:resize(self.opts.batchSize, self.opts.nhid)
  reprB:resize(self.opts.batchSize, self.opts.nhid)
  
  local batch_sentsA = {}
  local batch_sentsB = {}
  local sims = {}
  for i = 1, #sentsA do
    local sentA = sentsA[i]
    local sentB = sentsB[i]
    batch_sentsA[#batch_sentsA + 1] = UniText_Dataset.sent2ints(self.opts.src_vocab, sentA)
    batch_sentsB[#batch_sentsB + 1] = UniText_Dataset.sent2ints(self.opts.src_vocab, sentB)
    if i % self.opts.batchSize == 0 or i == #sentsA then
      local xa, xa_mask = self:toBatch(batch_sentsA, self.opts.src_vocab.EOS, 
        self.opts.batchSize, self.opts.src_vocab.UNK)
      local xb, xb_mask = self:toBatch(batch_sentsB, self.opts.src_vocab.EOS, 
        self.opts.batchSize, self.opts.src_vocab.UNK)
      
      reprA:copy( self.auto_encoder:fpropEncoder(xa, xa_mask) )
      reprB:copy( self.auto_encoder:fpropEncoder(xb, xb_mask) )
      
      local simMat = self:getCosSim(reprA, reprB)
      assert( simMat:size(1) == self.opts.batchSize )
      for j = 1, #batch_sentsA do
        sims[#sims + 1] = simMat[j]
      end
      
      batch_sentsA = {}
      batch_sentsB = {}
    end
  end
  
  assert(#sims == #sentsA, 'number of similarities should be the same as sentence pairs')
  
  return sims
end


function Scorer:getCosSim(m1, m2)
  assert(m1:dim() == 2 and m2:dim() == 2, 'm1 and m2 must be matrices')
  -- compute the dot product
  local dot = torch.bmm( m1:view(m1:size(1), 1, m1:size(2)), 
    m2:view(m2:size(1), m2:size(2), 1) ):squeeze()
  -- compute the norm
  local m1_norm = m1:norm(2, 2):squeeze()
  local m2_norm = m2:norm(2, 2):squeeze()
  
  -- the denominator might be zero!
  local function replaceZeros(dot, m_norm)
    local mask = m_norm:eq(0)
    dot[mask] = 0
    m_norm[mask] = 1e-10
  end
  replaceZeros(dot, m1_norm)
  replaceZeros(dot, m2_norm)
  
  return dot:cdiv(m1_norm):cdiv(m2_norm)
end


local function main()
  --[[
  local m1 = torch.Tensor({ {1, 2, 3}, {4, 5, 6}  })
  local m2 = torch.Tensor({ {1, 0, 3}, {0, 2.5, 3} })
  -- local m2 = torch.Tensor({ {0, 0, 0}, {0, 2.5, 3} })
  
  print(m1)
  print(m2)
  
  local scorer = SimilarityScorer()
  print( scorer:getCosSim(m1, m2) )
  --]]
  
  --[[
  local auto_enc_path = '/disk/scratch1/xingxing.zhang/seq2seq/sent_simple/encdec_rl/auto_encoder/model_0.001.256.2L.t7'
  local batch_size = 32
  local scorer = SimilarityScorer(auto_enc_path, batch_size)
  --]]
end


if not package.loaded['SimilarityScorer'] then
	main()
else
	print '[SimilarityScorer] loaded as package!'
end

