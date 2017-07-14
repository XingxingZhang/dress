
require '.'
require 'LMScorer'

local function main()
  local cmd = torch.CmdLine()
  cmd:option('--lm', '', '')
  cmd:option('--batchSize', 16, '')
  cmd:option('--infile', '', '')
  local args = cmd:parse(arg)
  print(args)
  
  local lmscorer = LMScorer(args.lm, args.batchSize)
  local fin = io.open(args.infile)
  local cnt = 0
  local all_sents = {}
  while true do
    local line = fin:read()
    if line == nil then break end
    all_sents[#all_sents + 1] = line
  end
  
  local NTest = #all_sents
  local sents = {}
  for i = 1, NTest do
    sents[#sents + 1] = all_sents[i]
  end
  local norm_probs = lmscorer:score(sents)
  for i = 1, #sents do
    xprintln('i = %d', i)
    print(sents[i])
    print(norm_probs[i])
    print '\n\n'
  end
  
end

main()

