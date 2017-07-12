
require '.'

--[[
-- let's start with a simple one, without considering the efficiency
local function createAttentionModel(opts)
  local s_i = nn.Identity()()
  local h_jm1 = nn.Identity()()
  local x_in = nn.JoinTable(2){s_i, h_jm1}
  local ha = nn.Linear(2*opts.nhid, opts.nhid)(x_in)
  local h = nn.Tanh()(ha)
  local o = nn.Linear(opts.nhid, 1)(h)
  
  return nn.gModule({s_i, h_jm1}, {o})
end
--]]

local function createAttentionModel(opts)
  local enc_hs = nn.Identity()()  -- size: bs x seqlen x nhid
  local dec_h = nn.Identity()()   -- size: bs x nhid
  -- general transformation: resulting size: bs x nhid x 1
  local dec_h_t = nn.View(opts.nhid, 1)( nn.Linear(opts.nhid, opts.nhid)(dec_h) )
  -- get attention: dot_encdec size: bs x seqlen x 1
  local dot_encdec = nn.MM()( {enc_hs, dec_h_t} )
  -- be careful x_mask is not handled currently!!!
  local attention = nn.SoftMax()( nn.Sum(3)(dot_encdec) )   -- size: bs x seqlen
  -- bs x seqlen x nhid MM bs x seqlen x 1
  local mout = nn.Sum(3)( nn.MM(true, false)( {enc_hs, nn.View(-1, 1):setNumInputDims(1)(attention)} ) )
  
  local model = nn.gModule({enc_hs, dec_h}, {mout})
  
  return model
end

local function main()
  --[[
  local opts = {nhid = 10}
  local att = createAttentionModel(opts)
  local bs = 10
  local s_i = torch.rand(bs, opts.nhid)
  local h_jm1 = torch.rand(bs, opts.nhid)
  print(s_i:size())
  print(s_i)
  print(h_jm1:size())
  print(h_jm1)
  local out = att:forward({s_i, h_jm1})
  print(out:size())
  print(out)
  --]]
  local opts = {nhid = 5}
  local att_model = createAttentionModel(opts)
  
  local enc_hs = torch.rand(5, 10, 5)
  local dec_h = torch.rand(5, 5)
  print(enc_hs:size())
  print(enc_hs)
  print(dec_h:size())
  print(dec_h)
  
  local out = att_model:forward({enc_hs, dec_h})
  print('==here is the result!!!==')
  print(out:size())
  print(out)
  
end

main()
