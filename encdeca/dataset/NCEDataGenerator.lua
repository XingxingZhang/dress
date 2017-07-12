
require 'alias_method'

local NCEDataGenerator = torch.class('NCEDataGenerator')

function NCEDataGenerator:__init(vocab, nneg, power, normalizeUNK)
  power = power or 1.0
  if normalizeUNK == nil then normalizeUNK = true end
  self.nneg = nneg
  
  self.unigramProbs = self:initUnigramProbs(vocab, power, normalizeUNK)
  self.size = self.unigramProbs:size(1)
  print('probs sum')
  print(self.unigramProbs:sum())
  print(self.unigramProbs:size(1))
  self.aliasMethod = AliasMethod(self.unigramProbs:totable())
end

function NCEDataGenerator:initUnigramProbs(vocab, power, normalizeUNK)
  print('power', power)
  print('normalizeUNK', normalizeUNK)
  
  local freqs = vocab.freqs
  local uniqUNK = vocab.uniqUNK
  local unkID = vocab.UNK
  local word2idx = vocab.word2idx
  local vocabSize = vocab.nvocab
  
  if normalizeUNK then freqs[unkID] = math.ceil( freqs[unkID] / uniqUNK ) end
  
  local ifreqs = torch.LongTensor(freqs)
  local pfreqs = ifreqs:double():pow(power)
  
  local uprobs = pfreqs:div( pfreqs:sum() )
  while uprobs:sum() ~= 1 do
    uprobs = pfreqs:div( pfreqs:sum() )
    print( uprobs:sum() )
    break
  end
  
  return uprobs
end

function NCEDataGenerator:getYNegProbs(y, useGPU)
  local probs = self.unigramProbs
  local nneg = self.nneg
  
  assert(y:dim() == 2)
  
  local y_neg = self.aliasMethod:drawBatch(y:size(1) * y:size(2) * nneg)
  
  local y_ = y:reshape(y:size(1) * y:size(2))
  -- I think I should do this earlier
  -- y_[y_:eq(0)] = 1
  local y_prob = probs:index(1, y_):reshape(y:size(1), y:size(2))
  local y_neg_prob = probs:index(1, y_neg):reshape(y:size(1), y:size(2), nneg)
  y_neg = y_neg:reshape(y:size(1), y:size(2), nneg)
  
  if useGPU then
    return y_neg:cuda(), y_prob:cuda(), y_neg_prob:cuda()
  else
    return y_neg, y_prob:float(), y_neg_prob:float()
  end
end
