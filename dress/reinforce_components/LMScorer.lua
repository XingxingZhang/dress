
require 'shortcut'
require 'LSTMLM'
require 'RawLM_Dataset'


local Scorer = torch.class('LMScorer')


function Scorer:__init(lm_path, batch_size)
  local state_path = lm_path:sub(1, -3) .. 'state.t7'
  self.opts = torch.load(state_path)
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
  self.lm = LSTMLM(self.opts)
  self.lm:load(lm_path)
  
  print 'create and load LSTM language model done!'
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


function Scorer:score(sents)
  -- assert(#sents % self.opts.batchSize == 0, 'MUST have sents of a batch')
  local batch_sents = {}
  local norm_probs = {}
  for i, sent in ipairs(sents) do
    batch_sents[#batch_sents + 1] = RawLM_Dataset.sent2ints(self.opts.vocab, sent)
    if i % self.opts.batchSize == 0 or i == #sents then
      local x, y = RawLM_Dataset.toBatch(batch_sents, self.opts.vocab.EOS, self.opts.batchSize)
      local logp, lens = self.lm:scoreBatch(x, y)
      for j = 1, #batch_sents do
        norm_probs[#norm_probs + 1] = lens[j] == 0 and 0 or torch.exp(logp[j] / lens[j])
      end
      batch_sents = {}
    end
  end
  
  return norm_probs
end


local function main()
  local cmd = torch.CmdLine()
  cmd:option('--lm', '', '')
  local args = cmd:parse(arg)
  print(args)
  local lm_path = args.lm
  local lmscorer = LMScorer(lm_path)
end

if not package.loaded['LMScorer'] then
	main()
else
	print '[LMScorer] loaded as package!'
end

