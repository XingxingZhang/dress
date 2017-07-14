
require 'shortcut'

local WordEmbed = torch.class('WordEmbeddingFT')

-- input should be torch7 file
function WordEmbed:__init(embedFile)
  self.embed, self.word2idx, self.idx2word = unpack( torch.load(embedFile) )
  xprintln('load embedding done!')
  self.lowerCase = true
  for _, word in ipairs(self.idx2word) do
    if word ~= word:lower() then
      self.lowerCase = false
    end
  end
  print('lower case: ')
  print(self.lowerCase)
end

function WordEmbed:releaseMemory()
  self.embed = nil
  self.word2idx = nil
  self.idx2word = nil
  collectgarbage()
end

function WordEmbed:initMatFT(mat, vocab, ftFactor)
  assert(mat:size(2) == self.embed:size(2))
  local mask = torch.ones(mat:size(1))
  
  local idx2word = vocab.idx2word
  local nvocab = #idx2word
  local cnt = 0
  for wid = 1, nvocab do
    local word = idx2word[wid]
    word = self.lowerCase and word:lower() or word
    local wid_ = self.word2idx[word]
    if wid_ ~= nil then
      mat[wid] = self.embed[wid_]
      cnt = cnt + 1
      mask[wid] = ftFactor
    -- else
    --  print(word)
    end
  end
  print(string.format('word embedding coverage: %d / %d = %f', cnt, nvocab, cnt / nvocab))
  
  return mask
end


