
require 'basic'
require 'EMaskedClassNLLCriterion'
require 'shortcut'

local model_utils = require 'model_utils'

local EncDec = torch.class('EncDec', 'BModel')

function EncDec:__init(opts)
  self.opts = opts
  self.name = 'EncDec'
  opts.nivocab, opts.novocab = opts.src_vocab.nvocab, opts.dst_vocab.nvocab
  self:createNetwork(opts)
  
  if opts.optimMethod == 'AdaGrad' then
    self.optimMethod = optim.adagrad
  elseif opts.optimMethod == 'Adam' then
    self.optimMethod = optim.adam
  elseif opts.optimMethod == 'AdaDelta' then
    self.optimMethod = optim.adadelta
  elseif opts.optimMethod == 'SGD' then
    self.optimMethod = optim.sgd
  end
end

function EncDec:transData(d)
  if self.opts.useGPU then 
    return d:cuda() 
  else
    return d
  end
end

function EncDec:createLSTM(x_t, c_tm1, h_tm1, nin, nhid)
  -- compute activations of four gates all together
  local x2h = nn.Linear(nin, nhid * 4)(x_t)
  local h2h = nn.Linear(nhid, nhid * 4)(h_tm1)
  local allGatesActs = nn.CAddTable()({x2h, h2h})
  local allGatesActsSplits = nn.SplitTable(2)( nn.Reshape(4, nhid)(allGatesActs) )
  -- unpack all gate activations
  local i_t = nn.Sigmoid()( nn.SelectTable(1)( allGatesActsSplits ) )
  local f_t = nn.Sigmoid()( nn.SelectTable(2)( allGatesActsSplits ) )
  local o_t = nn.Sigmoid()( nn.SelectTable(3)( allGatesActsSplits ) )
  local n_t = nn.Tanh()( nn.SelectTable(4)( allGatesActsSplits ) )
  -- compute new cell
  local c_t = nn.CAddTable()({
      nn.CMulTable()({ i_t, n_t }),
      nn.CMulTable()({ f_t, c_tm1 })
    })
  -- compute new hidden state
  local h_t = nn.CMulTable()({ o_t, nn.Tanh()( c_t ) })
  
  return c_t, h_t
end

function EncDec:createEncoder(opts)
  local emb = nn.LookupTable(opts.nivocab, opts.nin)
  local x_t = nn.Identity()()
  local s_tm1 = nn.Identity()()
  
  local in_t = { [0] = emb(x_t):annotate{name='enc_lookup'} }
  local s_t = {}
  local splits_tm1 = {s_tm1:split(2 * opts.nlayers)}
  
  for i = 1, opts.nlayers do
    local c_tm1_i = splits_tm1[i + i - 1]
    local h_tm1_i = splits_tm1[i + i]
    local x_t_i = in_t[i - 1]
    local c_t_i, h_t_i = nil, nil
    
    if opts.dropout > 0 then
      printf( 'lstm encoder layer %d, dropout = %f\n', i, opts.dropout) 
      x_t_i = nn.Dropout(opts.dropout)(x_t_i)
    end
    
    if i == 1 then
      c_t_i, h_t_i = self:createLSTM(x_t_i, c_tm1_i, h_tm1_i, opts.nin, opts.nhid)
    else
      c_t_i, h_t_i = self:createLSTM(x_t_i, c_tm1_i, h_tm1_i, opts.nhid, opts.nhid)
    end
    s_t[i+i-1] = c_t_i
    s_t[i+i] = h_t_i
    in_t[i] = h_t_i
  end
  
  local model = nn.gModule({x_t, s_tm1}, {nn.Identity()(s_t)})
  return self:transData(model)
end

function EncDec:createDecoder(opts)
  local emb = nn.LookupTable(opts.nivocab, opts.nin)
  local x_t = nn.Identity()()
  local s_tm1 = nn.Identity()()
  
  local in_t = { [0] = emb(x_t):annotate{name='dec_lookup'} }
  local s_t = {}
  local splits_tm1 = {s_tm1:split(2 * opts.nlayers)}
  
  for i = 1, opts.nlayers do
    local c_tm1_i = splits_tm1[i + i - 1]
    local h_tm1_i = splits_tm1[i + i]
    local x_t_i = in_t[i - 1]
    local c_t_i, h_t_i = nil, nil
    
    if opts.dropout > 0 then
      printf( 'lstm encoder layer %d, dropout = %f\n', i, opts.dropout) 
      x_t_i = nn.Dropout(opts.dropout)(x_t_i)
    end
    
    if i == 1 then
      c_t_i, h_t_i = self:createLSTM(x_t_i, c_tm1_i, h_tm1_i, opts.nin, opts.nhid)
    else
      c_t_i, h_t_i = self:createLSTM(x_t_i, c_tm1_i, h_tm1_i, opts.nhid, opts.nhid)
    end
    s_t[i+i-1] = c_t_i
    s_t[i+i] = h_t_i
    in_t[i] = h_t_i
  end
  local h_L = s_t[2*opts.nlayers]
  if opts.dropout > 0 then
    h_L = nn.Dropout(opts.dropout)(h_L)
    printf('apply dropout before output layer, drop = %f\n', opts.dropout)
  end
  local y_a = nn.Linear(opts.nhid, opts.novocab)(h_L)
  local y_prob = nn.LogSoftMax()(y_a)
  
  local model = nn.gModule({x_t, s_tm1}, {nn.Identity()(s_t), y_prob})
  return self:transData(model)
end

function EncDec:initModel(opts)
  self.enc_h0 = {}
  for i = 1, 2*opts.nlayers do
    self.enc_h0[i] = self:transData( torch.zeros(opts.batchSize, opts.nhid) )
  end
  print(self.enc_h0)
  
  self.enc_hs = {}   -- including all h_t and c_t
  for j = 0, opts.seqLen do
    self.enc_hs[j] = {}
    for d = 1, 2 * opts.nlayers do
      self.enc_hs[j][d] = self:transData( torch.zeros(opts.batchSize, opts.nhid) )
    end
  end
  
  self.df_enc_hs = {}
  
  self.dec_hs = {}
  for j = 0, opts.seqLen do
    self.dec_hs[j] = {}
    for d = 1, 2*opts.nlayers do
      self.dec_hs[j][d] = self:transData( torch.zeros(opts.batchSize, opts.nhid) )
    end
  end
  
  self.df_dec_hs = {}
  self.df_dec_h = {} -- remember always to reset it to zero!
  self.df_dec_h_cp = {}
  for i = 1, 2 * opts.nlayers do
    self.df_dec_h[i] = self:transData( torch.zeros(opts.batchSize, opts.nhid) )
    self.df_dec_h_cp[i] = self:transData( torch.zeros(opts.batchSize, opts.nhid) )
  end
  print(self.df_dec_h)
end

function EncDec:createNetwork()
  self.enc_lstm_master = self:createEncoder(self.opts)
  self.dec_lstm_master = self:createDecoder(self.opts)
  self.params, self.grads = model_utils.combine_all_parameters(self.enc_lstm_master, self.dec_lstm_master)
  self.params:uniform(-self.opts.initRange, self.opts.initRange)
  self:print( string.format( '#params %d', self.params:size(1) ) )
  self:print('clone encoder and decoder ...')
  self.enc_lstms = model_utils.clone_many_times(self.enc_lstm_master, self.opts.seqLen)
  self.dec_lstms = model_utils.clone_many_times(self.dec_lstm_master, self.opts.seqLen)
  self:print('clone encoder and decoder done!')
  
  self.criterions = {}
  for i = 1, self.opts.seqLen do
    self.criterions[i] = self:transData(EMaskedClassNLLCriterion())
  end
  
  self:initModel(self.opts)
end

function EncDec:trainBatch(x, x_mask, y, sgdParam)
  x = self:transData(x)
  x_mask = self:transData(x_mask)
  y = self:transData(y)
  --[[
  if x_mask:eq(0):sum() > 0  then
    print(x_mask)
  end
  --]]
  local xlen = x_mask:sum(1):view(-1)
  --[[
  print('x = ')
  print(x)
  print('x_mask = ')
  print(x_mask)
  print('y = ')
  print(y)
  --]]
  -- if y:eq(0):sum() > 0 then print(y) end
  -- set to training mode
  self.enc_lstm_master:training()
  self.dec_lstm_master:training()
  for i = 1, #self.enc_lstms do
    self.enc_lstms[i]:training()
  end
  for i = 1, #self.dec_lstms do
    self.dec_lstms[i]:training()
  end
  
  local function feval(params_)
    if self.params ~= params_ then
      self.params:copy(params_)
    end
    self.grads:zero()
    
    -- encoder forward pass
    for i = 1, 2 * self.opts.nlayers do
      self.enc_hs[0][i]:zero()
    end
    local Tx = x:size(1)
    for t = 1, Tx do
      self.enc_hs[t] = self.enc_lstms[t]:forward({x[{ t, {} }], self.enc_hs[t-1]})
      if self.opts.useGPU then cutorch.synchronize() end
    end
    -- decoder forward pass
    --[[
    for i = 1, 2 * self.opts.nlayers do
      self.dec_hs[0][i]:copy(self.enc_hs[Tx][i])
    end
    --]]
    for i = 1, 2 * self.opts.nlayers do
      for j = 1, self.opts.batchSize do
        self.dec_hs[0][i][{ j, {} }]:copy( self.enc_hs[ xlen[j] ][i][{ j, {} }] )
      end
    end
    
    local Ty = y:size(1) - 1
    local y_preds = {}
    local loss = 0
    for t = 1, Ty do
      self.dec_hs[t], y_preds[t] = unpack( self.dec_lstms[t]:forward({y[{ t, {} }], self.dec_hs[t-1]}) )
      local loss_ = self.criterions[t]:forward({y_preds[t], y[{t+1, {}}], self.opts.batchSize})
      loss = loss + loss_
      
      if self.opts.useGPU then cutorch.synchronize() end
    end
    
    -- decoder backward pass
    for i = 1, 2 * self.opts.nlayers do
      self.df_dec_h[i]:zero()
    end
    
    for t = Ty, 1, -1 do
      local df_crit = self.criterions[t]:backward({y_preds[t], y[{t+1, {}}], self.opts.batchSize})
      local tmp = self.dec_lstms[t]:backward({y[{ t, {}}], self.dec_hs[t-1]}, {self.df_dec_h, df_crit})[2]
      model_utils.copy_table(self.df_dec_h, tmp)
      
      if self.opts.useGPU then cutorch.synchronize() end
    end
    
    -- encoder backward pass
    model_utils.copy_table(self.df_dec_h_cp, self.df_dec_h)
    local min_xlen = xlen:min()
    for t = Tx, 1, -1 do
      if t >= min_xlen then
        for i = 1, self.opts.batchSize do
          if x_mask[{ t, i }] == 0 then
            for d = 1, 2 * self.opts.nlayers do
              self.df_dec_h[d][{ i, {} }]:zero()
            end
          elseif t == xlen[i] then
            for d = 1, 2 * self.opts.nlayers do
              self.df_dec_h[d][{ i, {} }]:copy( self.df_dec_h_cp[d][{ i, {} }] )
            end
          end
        end
      end
      
      local tmp = self.enc_lstms[t]:backward({x[{ t, {} }], self.enc_hs[t-1]}, self.df_dec_h)[2]
      model_utils.copy_table(self.df_dec_h, tmp)
      
      if self.opts.useGPU then cutorch.synchronize() end
    end
    
    if self.opts.gradClip < 0 then
      local clip = -self.opts.gradClip
      self.grads:clamp(-clip, clip)
    elseif self.opts.gradClip > 0 then
      local maxGradNorm = self.opts.gradClip
      local gradNorm = self.grads:norm()
      if gradNorm > maxGradNorm then
        local shrinkFactor = maxGradNorm / gradNorm
        self.grads:mul(shrinkFactor)
      end
    end
    
    return loss, self.grads
  end
  
  local _, loss_ = self.optimMethod(feval, self.params, sgdParam)
  return loss_[1]
end

function EncDec:validBatch(x, x_mask, y)
  x = self:transData(x)
  x_mask = self:transData(x_mask)
  y = self:transData(y)
  -- set to evaluation mode
  self.enc_lstm_master:evaluate()
  self.dec_lstm_master:evaluate()
  for i = 1, #self.enc_lstms do
    self.enc_lstms[i]:evaluate()
  end
  for i = 1, #self.dec_lstms do
    self.dec_lstms[i]:evaluate()
  end
  
  local xlen = x_mask:sum(1):view(-1)
  -- encoder forward pass
  for i = 1, 2 * self.opts.nlayers do
    self.enc_hs[0][i]:zero()
  end
  local Tx = x:size(1)
  for t = 1, Tx do
    self.enc_hs[t] = self.enc_lstms[t]:forward({x[{ t, {} }], self.enc_hs[t-1]})
    if self.opts.useGPU then cutorch.synchronize() end
  end
  -- decoder forward pass
  --[[
  for i = 1, 2 * self.opts.nlayers do
    self.dec_hs[0][i]:copy(self.enc_hs[Tx][i])
  end
  --]]
  for i = 1, 2 * self.opts.nlayers do
    for j = 1, self.opts.batchSize do
      self.dec_hs[0][i][{ j, {} }]:copy( self.enc_hs[ xlen[j] ][i][{ j, {} }] )
    end
  end
  
  local Ty = y:size(1) - 1
  local y_preds = {}
  local loss = 0
  for t = 1, Ty do
    self.dec_hs[t], y_preds[t] = unpack( self.dec_lstms[t]:forward({y[{ t, {} }], self.dec_hs[t-1]}) )
    local loss_ = self.criterions[t]:forward({y_preds[t], y[{t+1, {}}], self.opts.batchSize})
    loss = loss + loss_
    
    if self.opts.useGPU then cutorch.synchronize() end
  end
  
  return loss, y_preds
end





