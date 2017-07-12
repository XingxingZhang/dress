
require 'torch'
require 'nn'
require 'optim'
require 'nngraph'
require 'EMaskedClassNLLCriterion'
require 'NCE'
require 'NCEMaskedLoss'

require 'basic'

local model_utils = require 'model_utils'
local lstm_util = require 'lstmutil'

local LSTMLM = torch.class('LSTMNCELM', 'BModel')

local function transferData(useGPU, data)
  if useGPU then
    return data:cuda()
  else
    return data
  end
end

function LSTMLM:__init(opts)
  self.opts = opts
  self.name = 'LSTMNCELM'
  self:print( 'build LSTMNCELM ...' )
  opts.nivocab = opts.nivocab or opts.nvocab
  opts.novocab = opts.novocab or opts.nvocab
  opts.seqlen = opts.seqlen or 10
  self.nceZ = torch.exp(opts.lnZ)
  self:print(string.format('lnZ = %f, self.nceZ = %f', opts.lnZ, self.nceZ))
  
  self.lstm, self.nce, self.softmax = self:createNetwork(opts)
  self.params, self.grads = model_utils.combine_all_parameters(self.lstm, self.nce)
  self.params:uniform(-opts.initRange, opts.initRange)
  self:print( string.format('param size %d\n', self.params:size(1)) )
  print(self.params[{ {1, 10} }])
  
  local nce_module, _ = model_utils.share_nce_softmax(self.nce, self.softmax)
  if opts.learnZ then
    nce_module.bias:fill(self.nceZ)
  end
  
  self:print( 'Begin to clone model' )
  self.lstms = model_utils.clone_many_times(self.lstm, opts.seqLen)
  self:print( 'Clone model done!' )
  
  self:print('init states')
  self:setup(opts)
  self:print('init states done!')
  
  if opts.optimMethod == 'AdaGrad' then
    self.optimMethod = optim.adagrad
  elseif opts.optimMethod == 'Adam' then
    self.optimMethod = optim.adam
  elseif opts.optimMethod == 'AdaDelta' then
    self.optimMethod = optim.adadelta
  elseif opts.optimMethod == 'SGD' then
    self.optimMethod = optim.sgd
  end
  
  self:print( 'build LSTMNCELM done!' )
end

function LSTMLM:createNetwork(opts)
  local lstm = lstm_util.createDeepLSTM(opts)
  print 'done!'
  local function createNCE()
    local h_t = nn.Identity()()
    local y_t = nn.Identity()()
    local mask_t = nn.Identity()()
    local y_neg_t = nn.Identity()()
    local y_prob_t = nn.Identity()()
    local y_neg_prob_t = nn.Identity()()
    local div = nn.Identity()()
    
    local dropped = nn.Dropout( opts.dropout )( h_t )
    local nce_cost = NCE(opts.nhid, opts.novocab, self.nceZ, opts.learnZ)({dropped, y_t,
        y_neg_t, y_prob_t, y_neg_prob_t})
    local nce_loss = NCEMaskedLoss()({nce_cost, mask_t, div})
    local nce = nn.gModule({h_t, y_t, y_neg_t, y_prob_t, y_neg_prob_t, mask_t, div}, {nce_loss})
    
    return nce
  end
  
  local function createSoftmax()
    local h_t = nn.Identity()()
    local y_t = nn.Identity()()
    local div = nn.Identity()()
    
    local dropped = nn.Dropout( opts.dropout )( h_t )
    local h2y = nn.Linear(opts.nhid, opts.novocab)(dropped)
    local y_pred = nn.LogSoftMax()(h2y)
    local err = EMaskedClassNLLCriterion()({y_pred, y_t, div})
    local softmax = nn.gModule({h_t, y_t, div}, {err, y_pred})
    
    return softmax
  end
  
  local nce = createNCE()
  local softmax = createSoftmax()
  
  if opts.useGPU then
    nce = nce:cuda()
    softmax = softmax:cuda()
  end
  
  return lstm, nce, softmax
end

function LSTMLM:setup(opts)
  self.hidStates = {}   -- including all h_t and c_t
  self.initStates = {}
  self.df_hidStates = {}
  self.df_StatesT = {}
  
  for i = 1, 2*opts.nlayers do
    self.initStates[i] = transferData(opts.useGPU, torch.ones(opts.batchSize, opts.nhid) * opts.initHidVal)
    self.df_StatesT[i] = transferData(opts.useGPU, torch.zeros(opts.batchSize, opts.nhid))
  end
  -- self.hidStates[0] = self.initStates
  self.err = transferData(opts.useGPU, torch.zeros(opts.seqLen))
  self.hidL = opts.useGPU and torch.CudaTensor() or torch.Tensor()
  self.df_hidL = opts.useGPU and torch.CudaTensor() or torch.Tensor()
end

function LSTMLM:trainBatch(x, y, y_neg, y_prob, y_neg_prob, mask, sgdParam)
  if self.opts.useGPU then
    x = x:cuda()
    y = y:cuda()
    y_neg = y_neg:cuda()
    y_prob = y_prob:cuda()
    y_neg_prob = y_neg_prob:cuda()
    mask = mask:cuda()
  end
  
  local function feval(params_)
    if self.params ~= params_ then
      self.params:copy(params_)
    end
    self.grads:zero()
    
    ----------------------------------------
    ----- forward pass ------
    local T = x:size(1)
    local batchSize = x:size(2)
    self.hidL:resize(T * batchSize, self.opts.nhid)
    self.hidStates[0] = {}
    for i = 1, 2*self.opts.nlayers do
      self.hidStates[0][i] = self.initStates[i][{ {1, batchSize}, {} }]
    end
    for t = 1, T do
      local s_tm1 = self.hidStates[t - 1]
      self.hidStates[t] = self.lstms[t]:forward({ x[{ t, {} }], s_tm1 })
      self.hidL[{ {(t-1)*batchSize + 1, t*batchSize}, {} }] = self.hidStates[t][2*self.opts.nlayers]
    end
    
    -- now we've got the hidden states self.hiddenStates
    -- ready to compute the nce
    local y_ = y:view(y:size(1) * y:size(2))
    local allHiddenStates = self.hidL
    local y_neg_ = y_neg:view(y_neg:size(1) * y_neg:size(2), y_neg:size(3))
    local y_prob_ = y_prob:view(-1)
    local y_neg_prob_ = y_neg_prob:view(y_neg_prob:size(1) * y_neg_prob:size(2), y_neg_prob:size(3))
    local err = self.nce:forward({allHiddenStates, y_, y_neg_, y_prob_, y_neg_prob_, mask, batchSize})
    local loss = err
    ------------------------------------------
    
    ------------------------------------------
    ----- backward pass ---------
    local derr = transferData(self.opts.useGPU, torch.ones(1))
    
    local df_h_from_y, _, _, _, _, _, _ = unpack( self.nce:backward(
      {allHiddenStates, y_, y_neg_, y_prob_, y_neg_prob_, mask, batchSize}, 
      derr
      )
    )
    
    for i = 1, 2*self.opts.nlayers do
      self.df_StatesT[i]:zero()
    end
    self.df_hidStates[T] = self.df_StatesT
    
    for t = T, 1, -1 do
      -- printf('t = %d\n', t)
      local tmp = df_h_from_y[{ {(t-1)*batchSize + 1, t*batchSize}, {} }]
      self.df_hidStates[t][2*self.opts.nlayers]:add(tmp)
      
      local s_tm1 = self.hidStates[t - 1]
      -- local derr = transferData(self.opts.useGPU, torch.ones(1))
      local _, df_hidStates_tm1 = unpack(
        self.lstms[t]:backward(
          {x[{ t, {} }], s_tm1},
           self.df_hidStates[t]
          )
        )
      self.df_hidStates[t-1] = df_hidStates_tm1
      
      if self.opts.useGPU then
        cutorch.synchronize()
      end
    end
    
    -- clip the gradients
    -- self.grads:clamp(-5, 5)
    -- clip the gradients
    -- self.grads:clamp(-5, 5)
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

function LSTMLM:validBatch(x, y)
  if self.opts.useGPU then
    x = x:cuda()
    y = y:cuda()
  end
  
  ----------------------------------------
  ----- forward pass ------
  local T = x:size(1)
  local batchSize = x:size(2)
  self.hidL:resize(T * batchSize, self.opts.nhid)
  self.hidStates[0] = {}
  for i = 1, 2*self.opts.nlayers do
    self.hidStates[0][i] = self.initStates[i][{ {1, batchSize}, {} }]
  end
  
  -- self.hidStates[0] = self.
  for t = 1, T do
    local s_tm1 = self.hidStates[t - 1]
    self.hidStates[t] = self.lstms[t]:forward({ x[{ t, {} }], s_tm1 })
    self.hidL[{ {(t-1)*batchSize + 1, t*batchSize}, {} }] = self.hidStates[t][2*self.opts.nlayers]
  end
  
  -- now we've got the hidden states self.hiddenStates
  -- ready to compute the softmax
  
  local y_ = y:reshape(y:size(1) * y:size(2))
  local allHiddenStates = self.hidL
  local err, y_pred = unpack( self.softmax:forward({allHiddenStates, y_, batchSize}) )
  local loss = err
  
  return loss, y_pred
end

function LSTMLM:disableDropout()
  model_utils.disable_dropout( self.lstms )
  model_utils.disable_dropout( {self.softmax} )
end

function LSTMLM:enableDropout()
  model_utils.enable_dropout( self.lstms )
  model_utils.enable_dropout( {self.softmax} )
end



