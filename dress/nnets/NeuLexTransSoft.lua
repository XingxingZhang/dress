
require 'basic'
require 'LookupTable_ft'
require 'EMaskedClassNLLCriterion'

local model_utils = require 'model_utils'

local NeuLexTrans = torch.class('NeuLexTransSoft', 'BModel')

function NeuLexTrans:__init(opts)
  self.opts = opts
  self.name = 'NeuLexTransSoft'
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

function NeuLexTrans:transData(d)
  if self.opts.useGPU then
    return d:cuda() 
  else
    return d
  end
end

function NeuLexTrans:createLSTM(x_t, c_tm1, h_tm1, nin, nhid)
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
  
  if self.opts.recDropout and self.opts.recDropout > 0 then
    n_t = nn.Dropout(self.opts.recDropout)(n_t)
    printf( 'lstm, RECURRENT dropout = %f\n', self.opts.recDropout) 
  end
  
  -- compute new cell
  local c_t = nn.CAddTable()({
      nn.CMulTable()({ i_t, n_t }),
      nn.CMulTable()({ f_t, c_tm1 })
    })
  -- compute new hidden state
  local h_t = nn.CMulTable()({ o_t, nn.Tanh()( c_t ) })
  
  return c_t, h_t
end

function NeuLexTrans:createEncoder(opts)
  local emb = (opts.embedOption ~= nil and opts.embedOption == 'fineTune')
    and LookupTable_ft(opts.nivocab, opts.nin)
    or nn.LookupTable(opts.nivocab, opts.nin)
  -- local emb = nn.LookupTable(opts.nivocab, opts.nin)
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

function NeuLexTrans:createSoftmax(opts)
  local enc_hs = nn.Identity()()
  local attention = nn.Identity()()
  -- bs x seqlen x nhid MM bs x seqlen x 1
  local mout = nn.Sum(3)( nn.MM(true, false)( {enc_hs, nn.View(-1, 1):setNumInputDims(1)(attention)} ) ):annotate{name = 'h_last'}
  local input = mout
  if opts.dropout > 0 then
    input = nn.Dropout(opts.dropout)(input)
    printf('apply dropout before output layer, drop = %f\n', opts.dropout)
  end
  
  local y_a = nn.Linear(opts.nhid, opts.novocab)(input)
  local y_prob = nn.LogSoftMax()(y_a):annotate{name = 'y_softmax'}
  local model = nn.gModule({enc_hs, attention}, {y_prob})
  
  return self:transData(model)
end

function NeuLexTrans:initModel(opts)
  self.enc_hs = {}   -- including all h_t and c_t
  for j = 0, opts.seqLen do
    self.enc_hs[j] = {}
    for d = 1, 2 * opts.nlayers do
      self.enc_hs[j][d] = self:transData( torch.zeros(opts.batchSize, opts.nhid) )
    end
  end
  -- all the hidden states
  self.all_enc_hs = self:transData( torch.zeros( self.opts.batchSize * self.opts.seqLen, self.opts.nhid ) )
  self.df_all_enc_hs = self:transData( torch.zeros( self.opts.batchSize * self.opts.seqLen, self.opts.nhid ) )
  self.df_dec_h = {}
  for i = 1, 2 * opts.nlayers do
    self.df_dec_h[i] = self:transData( torch.zeros(opts.batchSize, opts.nhid) )
  end
end

function NeuLexTrans:createNetwork()
  self.enc_lstm_master = self:createEncoder(self.opts)
  self.dec_softmax_master = self:createSoftmax(self.opts)
  self.params, self.grads = model_utils.combine_all_parameters(self.enc_lstm_master, self.dec_softmax_master)
  self.params:uniform(-self.opts.initRange, self.opts.initRange)
  
  self.mod_map = BModel.get_module_map({self.enc_lstm_master, self.dec_softmax_master})
  -- use pretrained word embeddings
  if self.opts.wordEmbedding ~= nil and self.opts.wordEmbedding ~= ''  then
    local enc_embed = self.mod_map.enc_lookup
    self.enc_embed = enc_embed
    if self.opts.embedOption == 'init' then
      model_utils.load_embedding_init(enc_embed, self.opts.src_vocab, self.opts.wordEmbedding)
    elseif self.opts.embedOption == 'fineTune' then
      model_utils.load_embedding_fine_tune(enc_embed, self.opts.src_vocab, self.opts.wordEmbedding, self.opts.fineTuneFactor)
    else
      error('invalid option -- ' .. self.opts.embedOption)
    end
  end
  
  self:print( string.format( '#params %d', self.params:size(1) ) )
  
  self:print('clone encoder and decoder ...')
  self.enc_lstms = model_utils.clone_many_times_emb_ft(self.enc_lstm_master, self.opts.seqLen)
  self.dec_softmaxs = model_utils.clone_many_times(self.dec_softmax_master, self.opts.seqLen)
  self:print('clone encoder and decoder done!')
  
  self.criterions = {}
  for i = 1, self.opts.seqLen do
    self.criterions[i] = self:transData(EMaskedClassNLLCriterion())
  end
  
  self:initModel(self.opts)
end

function NeuLexTrans:trainBatch(x, x_mask, a, y, sgdParam)
  x = self:transData(x)
  x_mask = self:transData(x_mask)
  a = self:transData(a)
  y = self:transData(y)
  
  local Tx = x:size(1)
  local Ty = y:size(1) - 1
  self.enc_lstm_master:training()
  self.dec_softmax_master:training()
  for i = 1, Tx do
    self.enc_lstms[i]:training()
  end
  for i = 1, Ty do
    self.dec_softmaxs[i]:training()
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
    -- used for copying all encoder states
    self.all_enc_hs:resize( self.opts.batchSize, Tx, self.opts.nhid )
    for t = 1, Tx do
      self.enc_hs[t] = self.enc_lstms[t]:forward({x[{ t, {} }], self.enc_hs[t-1]})
      -- copy all encoder states
      self.all_enc_hs[{ {}, t, {} }] = self.enc_hs[t][2*self.opts.nlayers]
      
      if self.opts.useGPU then cutorch.synchronize() end
    end
    
    local loss = 0
    local y_preds = {}
    -- decoder softmax forward pass
    local dec_start = self.opts.decStart
    for t = dec_start, Ty do
      y_preds[t] = self.dec_softmaxs[t]:forward({ self.all_enc_hs, a[{ t, {}, {} }] })
      local loss_ = self.criterions[t]:forward({y_preds[t], y[{t+1, {}}], self.opts.batchSize})
      loss = loss + loss_
    end
    
    self.df_all_enc_hs:resize(self.opts.batchSize, Tx, self.opts.nhid):zero()
    -- decoder softmax backward pass
    for t = Ty, dec_start, -1 do
      local df_y_pred = self.criterions[t]:backward({y_preds[t], y[{t+1, {}}], self.opts.batchSize})
      local df_all_hids, _ = unpack( self.dec_softmaxs[t]:backward({self.all_enc_hs, a[{ t, {}, {} }]}, df_y_pred) )
      -- self.df_all_enc_hs:indexAdd(1, a_idxs[t], df_a_hid)
      self.df_all_enc_hs:add(df_all_hids)
    end
    
    -- encoder backward pass
    for i = 1, 2*self.opts.nlayers do
      self.df_dec_h[i]:zero()
    end
    for t = Tx, 1, -1 do
      self.df_dec_h[2*self.opts.nlayers]:add( self.df_all_enc_hs[{ {}, t, {} }] )
      -- apply mask
      -- mask should be used here
      local cmask = x_mask[{ t, {} }]:view(self.opts.batchSize, 1):expand(self.opts.batchSize, self.opts.nhid)
      for i = 1, 2*self.opts.nlayers do
        self.df_dec_h[i]:cmul( cmask )
      end
      local _, tmp = unpack( self.enc_lstms[t]:backward({x[{ t, {} }], self.enc_hs[t-1]}, self.df_dec_h) )
      model_utils.copy_table(self.df_dec_h, tmp)
    end
    
    if self.opts.embedOption ~= nil and self.opts.embedOption == 'fineTune' then
      self.enc_embed:applyGradMask()
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

function NeuLexTrans:validBatch(x, x_mask, a, y)
  x = self:transData(x)
  x_mask = self:transData(x_mask)
  a = self:transData(a)
  y = self:transData(y)
  
  local Tx = x:size(1)
  local Ty = y:size(1) - 1
  self.enc_lstm_master:evaluate()
  self.dec_softmax_master:evaluate()
  for i = 1, Tx do
    self.enc_lstms[i]:evaluate()
  end
  for i = 1, Ty do
    self.dec_softmaxs[i]:evaluate()
  end
  
  -- encoder forward pass
  for i = 1, 2 * self.opts.nlayers do
    self.enc_hs[0][i]:zero()
  end
  -- used for copying all encoder states
  self.all_enc_hs:resize( self.opts.batchSize, Tx, self.opts.nhid )
  for t = 1, Tx do
    self.enc_hs[t] = self.enc_lstms[t]:forward({x[{ t, {} }], self.enc_hs[t-1]})
    -- copy all encoder states
    self.all_enc_hs[{ {}, t, {} }] = self.enc_hs[t][2*self.opts.nlayers]
    
    if self.opts.useGPU then cutorch.synchronize() end
  end
  
  local loss = 0
  local y_preds = {}
  -- decoder softmax forward pass
  local dec_start = self.opts.decStart
  for t = dec_start, Ty do
    y_preds[t] = self.dec_softmaxs[t]:forward({ self.all_enc_hs, a[{ t, {}, {} }] })
    local loss_ = self.criterions[t]:forward({y_preds[t], y[{t+1, {}}], self.opts.batchSize})
    loss = loss + loss_
  end
  
  return loss, y_preds 
end

function NeuLexTrans:evaluateMode()
  self.enc_lstm_master:evaluate()
  self.dec_softmax_master:evaluate()
  for i = 1, self.opts.seqLen do
    self.enc_lstms[i]:evaluate()
    self.dec_softmaxs[i]:evaluate()
  end
end

function NeuLexTrans:fpropEncoder(x)
  x = self:transData(x)
  local Tx = x:size(1)
  
  -- encoder forward pass
  for i = 1, 2 * self.opts.nlayers do
    self.enc_hs[0][i]:zero()
  end
  -- used for copying all encoder states
  self.all_enc_hs:resize( self.opts.batchSize, Tx, self.opts.nhid )
  for t = 1, Tx do
    self.enc_hs[t] = self.enc_lstms[t]:forward({x[{ t, {} }], self.enc_hs[t-1]})
    -- copy all encoder states
    self.all_enc_hs[{ {}, t, {} }] = self.enc_hs[t][2*self.opts.nlayers]
    
    if self.opts.useGPU then cutorch.synchronize() end
  end
end

function NeuLexTrans:fpropTrans(a)
  return self.dec_softmax_master:forward({ self.all_enc_hs, a })
end



