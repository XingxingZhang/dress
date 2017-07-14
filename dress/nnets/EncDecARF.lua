
require 'basic'
require 'EMaskedClassNLLCriterion'
require 'shortcut'
require 'LookupTable_ft'
require 'ReinforceSampler'
require 'SARI'
require 'ReinforceCriterion'

local model_utils = require 'model_utils'

local EncDecA = torch.class('EncDecARF', 'BModel')

function EncDecA:__init(opts)
  self.opts = opts
  self.name = 'EncDecARF'
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

function EncDecA:transData(d)
  if self.opts.useGPU then
    return d:cuda() 
  else
    return d
  end
end

function EncDecA:createLSTM(x_t, c_tm1, h_tm1, nin, nhid)
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
    printf( 'lstm [%s], RECURRENT dropout = %f\n', label, self.opts.recDropout) 
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

function EncDecA:createEncoder(opts)
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

function EncDecA:createDecoder(opts)
  -- local emb = nn.LookupTable(opts.nivocab, opts.nin)
  -- local emb = nn.LookupTable(opts.novocab, opts.nin)
  local emb = (opts.embedOption ~= nil and opts.embedOption == 'fineTune') 
    and LookupTable_ft(opts.novocab, opts.nin) 
    or nn.LookupTable(opts.novocab, opts.nin)
    
  local x_t = nn.Identity()()
  local s_tm1 = nn.Identity()()
  local cxt_t = nn.Identity()()
  -- join word embedings of x_t and context vector
  local x_cxt_t = nn.JoinTable(2){emb(x_t):annotate{name='dec_lookup'}, cxt_t}
  local in_t = {[0] = x_cxt_t}
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

-- enc_hs (hidden states of encoder); size: bs x seqlen x nhid
-- dec_h (current hidden state of decoder); size: bs x nhid
function EncDecA:getEncVecByAtt(enc_hs, dec_h, mask, mask_sub, opts)
  --[[
  local enc_hs = nn.Identity()()  -- size: bs x seqlen x nhid
  local dec_h = nn.Identity()()   -- size: bs x nhid
  --]]
  -- general transformation / or just use simple dot: resulting size: bs x nhid x 1
  local dec_h_t
  if opts.attention == 'dot' then
    dec_h_t = nn.View(opts.nhid, 1):setNumInputDims(1)( dec_h )
    self:print('attention type is dot!')
  elseif opts.attention == 'general' then
    dec_h_t = nn.View(opts.nhid, 1):setNumInputDims(1)( nn.Linear(opts.nhid, opts.nhid)(dec_h) )
    self:print('attention type is general!')
  else
    error('attention should be "dot" or "general"')
  end
  --local dec_h_t = nn.View(opts.nhid, 1)( nn.Linear(opts.nhid, opts.nhid)(dec_h) )
  -- get attention: dot_encdec size: bs x seqlen x 1
  local dot_encdec = nn.MM()( {enc_hs, dec_h_t} )
  -- applying mask here
  local dot_encdec_tmp = nn.CAddTable()({ 
      nn.CMulTable()({ nn.Sum(3)(dot_encdec), 
          mask }),
      mask_sub
    })    -- size: bs x seqlen
  
  -- be careful x_mask is not handled currently!!!
  local attention = nn.SoftMax()( dot_encdec_tmp ):annotate{name='attention'}   -- size: bs x seqlen
  -- bs x seqlen x nhid MM bs x seqlen x 1
  local mout = nn.Sum(3)( nn.MM(true, false)( {enc_hs, nn.View(-1, 1):setNumInputDims(1)(attention)} ) )
  
  return mout
end

function EncDecA:createAttentionDecoder(opts)
  -- local emb = nn.LookupTable(opts.nivocab, opts.nin)
  -- local emb = nn.LookupTable(opts.novocab, opts.nin)
  local emb = (opts.embedOption ~= nil and opts.embedOption == 'fineTune') 
    and LookupTable_ft(opts.novocab, opts.nin) 
    or nn.LookupTable(opts.novocab, opts.nin)
  
  local x_t = nn.Identity()()       -- previous word
  local s_tm1 = nn.Identity()()     -- previous hidden states
  local h_hat_tm1 = nn.Identity()() -- the previous hidden state used to predict y (input feed)
  local enc_hs = nn.Identity()()    -- all hidden states of encoder
  -- add mask
  local mask = nn.Identity()()
  local mask_sub = nn.Identity()()
  
  -- join word embedings of x_t and previous hidden (input feed)
  local x_cxt_t = nn.JoinTable(2){emb(x_t):annotate{name='dec_lookup'}, h_hat_tm1}
  local in_t = {[0] = x_cxt_t}
  local s_t = {}
  local splits_tm1 = {s_tm1:split(2 * opts.nlayers)}
  
  for i = 1, opts.nlayers do
    local c_tm1_i = splits_tm1[i + i - 1]
    local h_tm1_i = splits_tm1[i + i]
    local x_t_i = in_t[i - 1]
    local c_t_i, h_t_i = nil, nil
  
    if opts.dropout > 0 then
      printf( 'lstm decoder layer %d, dropout = %f\n', i, opts.dropout)
      x_t_i = nn.Dropout(opts.dropout)(x_t_i)
    end
    
    if i == 1 then
      c_t_i, h_t_i = self:createLSTM(x_t_i, c_tm1_i, h_tm1_i, opts.nin + opts.nhid, opts.nhid)
    else
      c_t_i, h_t_i = self:createLSTM(x_t_i, c_tm1_i, h_tm1_i, opts.nhid, opts.nhid)
    end
    s_t[i+i-1] = c_t_i
    s_t[i+i] = h_t_i
    in_t[i] = h_t_i
  end
  local h_L = s_t[2*opts.nlayers]
  -- now time for attention!
  local c_t = self:getEncVecByAtt(enc_hs, h_L, mask, mask_sub, opts)
  local h_hat_t_in = nn.JoinTable(2){h_L, c_t}
  local h_hat_t = nn.Tanh()( nn.Linear( 2*opts.nhid, opts.nhid )( h_hat_t_in ) ):annotate{name = 'h_last'}
  
  local h_hat_t_out = h_hat_t
  if opts.dropout > 0 then
    h_hat_t_out = nn.Dropout(opts.dropout)(h_hat_t_out)
    printf('apply dropout before output layer, drop = %f\n', opts.dropout)
  end
  
  local y_a = nn.Linear(opts.nhid, opts.novocab)(h_hat_t_out)
  local y_prob = nn.LogSoftMax()(y_a):annotate{name = 'y_softmax'}
  local out_sample = nn.ReinforceSampler('multinomial', false)(y_prob)
  
  local model = nn.gModule({x_t, s_tm1, h_hat_tm1, enc_hs, mask, mask_sub}, 
    {nn.Identity()(s_t), y_prob, h_hat_t, out_sample})
  
  return self:transData(model)
end


function EncDecA:initModel(opts)
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
  
  -- for attention model
  self.all_enc_hs = self:transData( torch.zeros( self.opts.batchSize * self.opts.seqLen, self.opts.nhid ) )
  self.df_all_enc_hs = self:transData( torch.zeros( self.opts.batchSize, self.opts.seqLen, self.opts.nhid ) )
  
  -- h_hat_t
  self.dec_hs_hat = {}
  for j = 0, opts.seqLen do
    self.dec_hs_hat[j] = self:transData( torch.zeros(opts.batchSize, opts.nhid) )
  end
  
  self.df_dec_h_hat = self:transData( torch.zeros(opts.batchSize, opts.nhid) )
  
  -- the reward is for reinforcement learning!
  self.reward = self:transData( torch.zeros(self.opts.seqLen, self.opts.batchSize) )
  self.reward_mask = self:transData( torch.zeros(self.opts.seqLen, self.opts.batchSize) )
  -- self.df_rf_sample = self:transData( torch.zeros(self.opts.batchSize, 1) )
  self.df_preds = self:transData( torch.Tensor(self.opts.batchSize, self.opts.novocab) )
  
  self.dummy_df_y_rf = self:transData( torch.zeros(self.opts.seqLen, self.opts.batchSize) )
end


function EncDecA:createNetwork()
  self.enc_lstm_master = self:createEncoder(self.opts)
  self.dec_lstm_master = self:createAttentionDecoder(self.opts)
  local all_modules = nn.Parallel(1, 1):add(self.enc_lstm_master):add(self.dec_lstm_master)
  self.cum_reward_predictors = {}
  for i = 1, self.opts.seqLen do
    self.cum_reward_predictors[i] = self:transData(nn.Linear(self.opts.nhid, 1))
    all_modules:add( self.cum_reward_predictors[i] )
  end
  self.params, self.grads = all_modules:getParameters()
  -- self.params, self.grads = model_utils.combine_all_parameters(self.enc_lstm_master, self.dec_lstm_master)
  self.params:uniform(-self.opts.initRange, self.opts.initRange)
  for i = 1, self.opts.seqLen do
    self.cum_reward_predictors[i].bias:fill(0.01)
    self.cum_reward_predictors[i].weight:fill(0)
  end
  self:print('combine and init model done!')
  -- print(self.params[{ {-257, -1} }])
  
  self.mod_map = BModel.get_module_map({self.enc_lstm_master, self.dec_lstm_master})
  -- use pretrained word embeddings
  if self.opts.wordEmbedding ~= nil and self.opts.wordEmbedding ~= ''  then
    local enc_embed = self.mod_map.enc_lookup
    local dec_embed = self.mod_map.dec_lookup
    self.enc_embed = enc_embed
    self.dec_embed = dec_embed
    if self.opts.embedOption == 'init' then
      model_utils.load_embedding_init(enc_embed, self.opts.src_vocab, self.opts.wordEmbedding)
      model_utils.load_embedding_init(dec_embed, self.opts.dst_vocab, self.opts.wordEmbedding)
    elseif self.opts.embedOption == 'fineTune' then
      model_utils.load_embedding_fine_tune(enc_embed, self.opts.src_vocab, self.opts.wordEmbedding, self.opts.fineTuneFactor)
      model_utils.load_embedding_fine_tune(dec_embed, self.opts.dst_vocab, self.opts.wordEmbedding, self.opts.fineTuneFactor)
    else
      error('invalid option -- ' .. self.opts.embedOption)
    end
  end
  
  self:print( string.format( '#params %d', self.params:size(1) ) )
  
  
  self:print('clone encoder and decoder ...')
  self.enc_lstms = model_utils.clone_many_times_emb_ft(self.enc_lstm_master, self.opts.seqLen)
  self.dec_lstms = model_utils.clone_many_times_emb_ft(self.dec_lstm_master, self.opts.seqLen)
  self:print('clone encoder and decoder done!')
  
  self.criterions = {}
  for i = 1, self.opts.seqLen do
    self.criterions[i] = self:transData(EMaskedClassNLLCriterion())
  end
  
  self.sampleStart = 3
  -- modified the implmentation of ReinforceCriterion
  self.rf_criterion = self:transData( nn.ReinforceCriterion(self.sampleStart, self.opts.seqLen, self.opts.batchSize) )
  
  self:initModel(self.opts)
end


function EncDecA:setSampleStart(start)
  self.sampleStart = start
end


function EncDecA:trainBatch(x, x_mask, y, sgdParam)
  x = self:transData(x)
  x_mask = self:transData(x_mask)
  y = self:transData(y)
  
  local xlen = x_mask:sum(1):view(-1)
  
  local x_mask_t = x_mask:t()
  local x_mask_sub = (-x_mask_t + 1) * -50
  x_mask_sub = self:transData( x_mask_sub )
  
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
  
  self.rf_criterion:setSampleStart(self.sampleStart)
  local train_batch_info = {}
  
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
    -- used for copying all encoder states
    self.all_enc_hs:resize( self.opts.batchSize * Tx, self.opts.nhid )
    local raw_idxs = self:transData( torch.linspace(0, Tx*(self.opts.batchSize-1), self.opts.batchSize):long() )
    for t = 1, Tx do
      self.enc_hs[t] = self.enc_lstms[t]:forward({x[{ t, {} }], self.enc_hs[t-1]})
      -- copy all encoder states
      local idxs = raw_idxs + t
      self.all_enc_hs:indexCopy(1, idxs, self.enc_hs[t][2*self.opts.nlayers])
      
      if self.opts.useGPU then cutorch.synchronize() end
    end
    local all_enc_hs = self.all_enc_hs:view(self.opts.batchSize, Tx, self.opts.nhid)
    
    -- decoder forward pass 
    for i = 1, 2 * self.opts.nlayers do
      for j = 1, self.opts.batchSize do
        self.dec_hs[0][i][{ j, {} }]:copy( self.enc_hs[ xlen[j] ][i][{ j, {} }] )
      end
    end
    
    -- ** This is the place where I need to use reinforcement learning ** --
    local Ty = y:size(1) - 1
    local Ty_rf = math.min( torch.round(y:size(1) * 1.2), self.opts.seqLen )
    local y_rf = self:transData( torch.LongTensor(Ty_rf, y:size(2)) )
    y_rf[{ 1, {} }] = y[{ 1, {} }]
    
    local y_preds = {}
    local nll_loss = 0
    local expected_rewards = {}
    local maxTy = self.sampleStart > Ty and Ty or Ty_rf - 1
    for t = 1, maxTy do
      local rf_sample
      self.dec_hs[t], y_preds[t], self.dec_hs_hat[t], rf_sample = unpack( 
        self.dec_lstms[t]:forward( { y_rf[{ t, {} }], self.dec_hs[t-1], self.dec_hs_hat[t-1], all_enc_hs, x_mask_t, x_mask_sub } ) 
      )
      
      if t < self.sampleStart then
        y_rf[{ t+1, {} }] = y[{ t+1, {} }]
        local loss_ = self.criterions[t]:forward({y_preds[t], y[{t+1, {}}], self.opts.batchSize})
        nll_loss = nll_loss + loss_
      else
        y_rf[{ t+1, {} }] = rf_sample:squeeze()
        expected_rewards[t] = self.cum_reward_predictors[t]:forward(self.dec_hs_hat[t])
      end
    end
    
    local rf_loss, n_seq, df_y_rf, df_expected_rewards
    if self.sampleStart <= Ty then
      self.reward:resize(y_rf:size()):zero()
      self.reward_mask:resize(y_rf:size()):zero()
      -- reinforce criterion in Action
      SARI.getDynBatch(x, x_mask, y, y_rf, self.reward, self.reward_mask, self.opts.src_vocab, self.opts.dst_vocab)
      -- rf_sample, exp_reward, true_reward, reward_mask
      rf_loss, n_seq = self.rf_criterion:forward({y_rf, expected_rewards, self.reward, self.reward_mask})
      df_y_rf, df_expected_rewards = unpack( self.rf_criterion:backward({y_rf, expected_rewards, self.reward, self.reward_mask}) )
    else
      rf_loss, n_seq = 0, 0
      self.dummy_df_y_rf:resize(y_rf:size()):zero()
      df_y_rf = self.dummy_df_y_rf
    end
    
    --[[
    xprintln('backward pass for rf_criterion')
    print(df_y_rf)
    print(df_expected_rewards[self.sampleStart])
    --]]
    -- xprintln('rf_loss = %f', rf_loss)
    
    -- decoder backward pass
    for i = 1, 2 * self.opts.nlayers do
      self.df_dec_h[i]:zero()
    end
    self.df_dec_h_hat:zero()
    -- initialize it at initModel
    self.df_all_enc_hs:resize( self.opts.batchSize, Tx, self.opts.nhid )
    self.df_all_enc_hs:zero()
    
    for t = maxTy, 1, -1 do
      local tmp_dec_h, tmp_dec_h_hat, tmp_all_enc_hs
      local df_y_rf_t = df_y_rf[{ t+1, {} }]:view( df_y_rf[{ t+1, {} }]:size(1), 1 )
      if t < self.sampleStart then
        local df_crit = self.criterions[t]:backward({y_preds[t], y[{t+1, {}}], self.opts.batchSize})
        local _, tmp_dec_h_, tmp_dec_h_hat_, tmp_all_enc_hs_, _, _ = unpack( self.dec_lstms[t]:backward(
          { y_rf[{ t, {} }], self.dec_hs[t-1], self.dec_hs_hat[t-1], all_enc_hs, x_mask_t, x_mask_sub }, 
          { self.df_dec_h, df_crit, self.df_dec_h_hat, df_y_rf_t })
        )
        tmp_dec_h, tmp_dec_h_hat, tmp_all_enc_hs = tmp_dec_h_, tmp_dec_h_hat_, tmp_all_enc_hs_
      else
        local dec_hs_hat_t = self.cum_reward_predictors[t]:backward(self.dec_hs_hat[t], df_expected_rewards[t])
        local df_crit = self.df_preds:resizeAs(y_preds[t]):zero()
        local _, tmp_dec_h_, tmp_dec_h_hat_, tmp_all_enc_hs_, _, _ = unpack( self.dec_lstms[t]:backward(
          { y_rf[{ t, {} }], self.dec_hs[t-1], self.dec_hs_hat[t-1], all_enc_hs, x_mask_t, x_mask_sub }, 
          { self.df_dec_h, df_crit, self.df_dec_h_hat, df_y_rf_t })
        )
        tmp_dec_h, tmp_dec_h_hat, tmp_all_enc_hs = tmp_dec_h_, tmp_dec_h_hat_, tmp_all_enc_hs_
      end
      
      model_utils.copy_table(self.df_dec_h, tmp_dec_h)
      self.df_dec_h_hat:copy(tmp_dec_h_hat)
      self.df_all_enc_hs:add(tmp_all_enc_hs)
      
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
            -- clear back prop error in padded place
            self.df_all_enc_hs[{ i, t, {} }]:zero()
          elseif t == xlen[i] then
            for d = 1, 2 * self.opts.nlayers do
              self.df_dec_h[d][{ i, {} }]:copy( self.df_dec_h_cp[d][{ i, {} }] )
            end
          end
        end
      end
      
      -- add errors to last layer of self.df_dec_h
      self.df_dec_h[2*self.opts.nlayers]:add( self.df_all_enc_hs[{ {}, t, {} }] )
      local tmp = self.enc_lstms[t]:backward({x[{ t, {} }], self.enc_hs[t-1]}, self.df_dec_h)[2]
      model_utils.copy_table(self.df_dec_h, tmp)
      
      if self.opts.useGPU then cutorch.synchronize() end
    end
    
    if self.opts.embedOption ~= nil and self.opts.embedOption == 'fineTune' then
      self.enc_embed:applyGradMask()
      self.dec_embed:applyGradMask()
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
    
    train_batch_info = {nll_loss = nll_loss, rf_loss = rf_loss, n_seq = n_seq}
    
    return nll_loss + rf_loss, self.grads
  end
  
  local _, loss_ = self.optimMethod(feval, self.params, sgdParam)
  return train_batch_info
end

function EncDecA:validBatch(x, x_mask, y)
  --[[
  x = self:transData(x)
  x_mask = self:transData(x_mask)
  y = self:transData(y)
  --]]
  x = self:transData(x)
  x_mask = self:transData(x_mask)
  y = self:transData(y)
  
  local xlen = x_mask:sum(1):view(-1)
  
  local x_mask_t = x_mask:t()
  local x_mask_sub = (-x_mask_t + 1) * -50
  x_mask_sub = self:transData( x_mask_sub )
  
  -- set to evaluation mode
  self.enc_lstm_master:evaluate()
  self.dec_lstm_master:evaluate()
  for i = 1, #self.enc_lstms do
    self.enc_lstms[i]:evaluate()
  end
  for i = 1, #self.dec_lstms do
    self.dec_lstms[i]:evaluate()
  end
  
  -- encoder forward pass
  for i = 1, 2 * self.opts.nlayers do
    self.enc_hs[0][i]:zero()
  end
  local Tx = x:size(1)
  -- used for copying all encoder states
  self.all_enc_hs:resize( self.opts.batchSize * Tx, self.opts.nhid )
  local raw_idxs = self:transData( torch.linspace(0, Tx*(self.opts.batchSize-1), self.opts.batchSize):long() )
  for t = 1, Tx do
    self.enc_hs[t] = self.enc_lstms[t]:forward({x[{ t, {} }], self.enc_hs[t-1]})
    -- copy all encoder states
    local idxs = raw_idxs + t
    self.all_enc_hs:indexCopy(1, idxs, self.enc_hs[t][2*self.opts.nlayers])
    
    if self.opts.useGPU then cutorch.synchronize() end
  end
  local all_enc_hs = self.all_enc_hs:view(self.opts.batchSize, Tx, self.opts.nhid)
  
  -- decoder forward pass
  for i = 1, 2 * self.opts.nlayers do
    for j = 1, self.opts.batchSize do
      self.dec_hs[0][i][{ j, {} }]:copy( self.enc_hs[ xlen[j] ][i][{ j, {} }] )
    end
  end
  
  local Ty = y:size(1) - 1
  local y_preds = {}
  local loss = 0
  for t = 1, Ty do
    -- self.dec_hs[t], y_preds[t] = unpack( self.dec_lstms[t]:forward({y[{ t, {} }], self.dec_hs[t-1]}) )
    self.dec_hs[t], y_preds[t], self.dec_hs_hat[t] = unpack( 
      self.dec_lstms[t]:forward( { y[{ t, {} }], self.dec_hs[t-1], self.dec_hs_hat[t-1], all_enc_hs, x_mask_t, x_mask_sub } ) 
      )
    local loss_ = self.criterions[t]:forward({y_preds[t], y[{t+1, {}}], self.opts.batchSize})
    loss = loss + loss_
    
    if self.opts.useGPU then cutorch.synchronize() end
  end
  
  return loss, y_preds
end


function EncDecA:validBatchSample(x, x_mask, y, greedy)
  x = self:transData(x)
  x_mask = self:transData(x_mask)
  y = self:transData(y)
  
  local xlen = x_mask:sum(1):view(-1)
  
  local x_mask_t = x_mask:t()
  local x_mask_sub = (-x_mask_t + 1) * -50
  x_mask_sub = self:transData( x_mask_sub )
  
  self.enc_lstm_master:evaluate()
  self.dec_lstm_master:evaluate()
  for i = 1, #self.enc_lstms do
    self.enc_lstms[i]:evaluate()
  end
  for i = 1, #self.dec_lstms do
    self.dec_lstms[i]:evaluate()
  end
  
  local sampleStart = 1
  self.rf_criterion:setSampleStart(sampleStart)
  local valid_batch_info = {}
  
  -- encoder forward pass
  for i = 1, 2 * self.opts.nlayers do
    self.enc_hs[0][i]:zero()
  end
  local Tx = x:size(1)
  -- used for copying all encoder states
  self.all_enc_hs:resize( self.opts.batchSize * Tx, self.opts.nhid )
  local raw_idxs = self:transData( torch.linspace(0, Tx*(self.opts.batchSize-1), self.opts.batchSize):long() )
  for t = 1, Tx do
    self.enc_hs[t] = self.enc_lstms[t]:forward({x[{ t, {} }], self.enc_hs[t-1]})
    -- copy all encoder states
    local idxs = raw_idxs + t
    self.all_enc_hs:indexCopy(1, idxs, self.enc_hs[t][2*self.opts.nlayers])
    
    if self.opts.useGPU then cutorch.synchronize() end
  end
  local all_enc_hs = self.all_enc_hs:view(self.opts.batchSize, Tx, self.opts.nhid)
  
  -- decoder forward pass 
  for i = 1, 2 * self.opts.nlayers do
    for j = 1, self.opts.batchSize do
      self.dec_hs[0][i][{ j, {} }]:copy( self.enc_hs[ xlen[j] ][i][{ j, {} }] )
    end
  end
  
  -- ** This is the place where I need to use reinforcement learning ** --
  local Ty = y:size(1) - 1
  local Ty_rf = math.min( torch.round(y:size(1) * 1.2), self.opts.seqLen )
  local y_rf = self:transData( torch.LongTensor(Ty_rf, y:size(2)) )
  y_rf[{ 1, {} }] = y[{ 1, {} }]
  
  local y_preds = {}
  local nll_loss = 0
  local expected_rewards = {}
  local maxTy = Ty_rf - 1
  for t = 1, maxTy do
    local rf_sample
    self.dec_hs[t], y_preds[t], self.dec_hs_hat[t], rf_sample = unpack( 
      self.dec_lstms[t]:forward( { y_rf[{ t, {} }], self.dec_hs[t-1], self.dec_hs_hat[t-1], all_enc_hs, x_mask_t, x_mask_sub } ) 
    )
    
    if not greedy then
      y_rf[{ t+1, {} }] = rf_sample:squeeze()
    else
      local maxv, maxi = y_preds[t]:max(2)
      y_rf[{ t+1, {} }] = maxi:squeeze()
    end
    expected_rewards[t] = self.cum_reward_predictors[t]:forward(self.dec_hs_hat[t])
  end
  
  self.reward:resize(y_rf:size()):zero()
  self.reward_mask:resize(y_rf:size()):zero()
  -- get reward
  SARI.getDynBatch(x, x_mask, y, y_rf, self.reward, self.reward_mask, self.opts.src_vocab, self.opts.dst_vocab)
  
  local size = self.reward:sum(1):ne(0):sum()
  return self.reward:sum(), size
end


function EncDecA:evaluateMode()
  -- set to evaluation mode
  self.enc_lstm_master:evaluate()
  self.dec_lstm_master:evaluate()
  for i = 1, #self.enc_lstms do
    self.enc_lstms[i]:evaluate()
  end
  for i = 1, #self.dec_lstms do
    self.dec_lstms[i]:evaluate()
  end
end




