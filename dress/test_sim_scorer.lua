
require '.'
require 'SimilarityScorer'


local function getCosSim(m1, m2)
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


local function test_sim()
  local auto_enc_path = '/disk/scratch1/xingxing.zhang/seq2seq/sent_simple/encdec_rl/auto_encoder/model_0.001.256.2L.t7'
  local batch_size = 32
  local scorer = SimilarityScorer(auto_enc_path, batch_size)
  
  local train_file = '/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/newsela/all/V0V4_V1V4_V2V4_V3V4_V0V3_V0V2_V1V3.aner.train'
  
  local fin_src = io.open(train_file .. '.src')
  local fin_dst = io.open(train_file .. '.dst')
  
  local srcs = {}
  local dsts = {}
  while true do
    local src = fin_src:read()
    local dst = fin_dst:read()
    
    if src == nil then break end
    
    srcs[#srcs + 1] = src
    dsts[#dsts + 1] = dst
  end
  
  local reprA = torch.CudaTensor()
  local reprB = torch.CudaTensor()
  local sims = scorer:score(srcs, dsts, reprA, reprB)
  
  for i = 1, #srcs do
    xprintln('i = %d', i)
    xprintln('src = %s', srcs[i])
    xprintln('dst = %s', dsts[i])
    xprintln('sim = %f\n\n', sims[i])
  end
  
  fin_src:close()
  fin_dst:close()
end


local function main()
  --[[
  local auto_enc_path = '/disk/scratch1/xingxing.zhang/seq2seq/sent_simple/encdec_rl/auto_encoder/model_0.001.256.2L.t7'
  local batch_size = 32
  local scorer = SimilarityScorer(auto_enc_path, batch_size)
  --]]
  
  --[[
  local m1 = torch.Tensor({ {1, 2, 3}, {4, 5, 6}  })
  -- local m2 = torch.Tensor({ {1, 0, 3}, {0, 2.5, 3} })
  local m2 = torch.Tensor({ {0, 0, 0}, {0, 2.5, 3} })
  
  print(m1)
  print(m2)
  
  print( getCosSim(m1, m2) )
  --]]
  
  test_sim()
end

main()
