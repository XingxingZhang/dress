
require 'basic'

local NeuDynFeat = torch.class('NeuDynFeatWeighter', 'BModel')

function NeuDynFeat:__init(opts)
  self.opts = opts
   self.name = 'NeuDynFeatWeighter'
  self:createNetwork()
  
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


function NeuDynFeat:transData(d)
  if self.opts.useGPU then
    return d:cuda()
  else
    return d
  end
end


function NeuDynFeat:createSoftmaxGate(opts)
  local nfeat = opts.nfeat
	local nhids = opts.nhids
	local nghid = opts.nghid
	local inDropout = opts.inDropout
	local dropout = opts.dropout
  
	local hs = {} 	-- hidden states
	local ys = {}	-- log p(y|x)
	local nhid = 0
	for i = 1, nfeat do
		hs[i] = nn.Identity()()
		ys[i] = nn.Identity()()
		nhid = nhid + nhids[i]
	end
	local h = nn.JoinTable(2)(hs)
	if inDropout > 0 then
		h = nn.Dropout(inDropout)(h)
	end
	local g_h = nn.Linear(nhid, nghid)(h)
  g_h = nn.ReLU()(g_h)
	if dropout > 0 then
		g_h = nn.Dropout(dropout)(g_h)
	end
	local g = nn.SoftMax()( nn.Linear(nghid, nfeat)(g_h) )
				:annotate{name = 'feat_weight'}
	local g_mm = nn.View(1, nfeat):setNumInputDims(1)(g)
	-- merge all log p(y|x) and convert p(y')
	local y = nn.Exp()( nn.JoinTable(2)(ys) )
	local y_mm = nn.View(nfeat, -1):setNumInputDims(1)(y)
	local y_prob = nn.Sum(2)( nn.MM(false, false){g_mm, y_mm} )
	local log_y = nn.Log()(y_prob)
  
	local input = {}
	for i = 1, nfeat do
		input[#input + 1] = hs[i]
		input[#input + 1] = ys[i]
	end
  
	local model = nn.gModule(input, {log_y})
	return self:transData(model)
end


function NeuDynFeat:createDeepSoftmaxGate(opts)
  local nfeat = opts.nfeat
	local nhids = opts.nhids
	local nghid = opts.nghid
	local inDropout = opts.inDropout
	local dropout = opts.dropout
  
	local hs = {} 	-- hidden states
	local ys = {}	-- log p(y|x)
	local nhid = 0
	for i = 1, nfeat do
		hs[i] = nn.Identity()()
		ys[i] = nn.Identity()()
		nhid = nhid + nhids[i]
	end
	local h = nn.JoinTable(2)(hs)
	if inDropout > 0 then
		h = nn.Dropout(inDropout)(h)
	end
  
  local g_out = {}
  for i = 1, nfeat do
    local g_h = nn.Linear(nhid, nghid)(h)
    g_h = nn.ReLU()(g_h)
    if dropout > 0 then
      g_h = nn.Dropout(dropout)(g_h)
    end
    g_out[#g_out + 1] = nn.Linear(nghid, 1)(g_h)
  end
  local g = nn.SoftMax()( nn.JoinTable(2)(g_out) )
          :annotate{name = 'feat_weight'}
  
	local g_mm = nn.View(1, nfeat):setNumInputDims(1)(g)
	-- merge all log p(y|x) and convert p(y')
	local y = nn.Exp()( nn.JoinTable(2)(ys) )
	local y_mm = nn.View(nfeat, -1):setNumInputDims(1)(y)
	local y_prob = nn.Sum(2)( nn.MM(false, false){g_mm, y_mm} )
	local log_y = nn.Log()(y_prob)
  
	local input = {}
	for i = 1, nfeat do
		input[#input + 1] = hs[i]
		input[#input + 1] = ys[i]
	end
  
	local model = nn.gModule(input, {log_y})
	return self:transData(model)
end


function NeuDynFeat:createNetwork()
  if self.opts.deepGate then
    self:print('using deep gate!')
    self.soft_gate = self:createDeepSoftmaxGate(self.opts)
  else
    self.print('using shallow gate!')
    self.soft_gate = self:createSoftmaxGate(self.opts)
  end
  
  self.params, self.grads = self.soft_gate:getParameters()
  self.params:uniform(-self.opts.initRange, self.opts.initRange)
  
  self.mod_map = BModel.get_module_map(self.soft_gate)
  print(table.keys(self.mod_map))
  print(self.mod_map.feat_weight)
  
  self.criterion = self:transData( EMaskedClassNLLCriterion() )
  self:print('create network done!')
end


function NeuDynFeat:addFeature(name, feat)
  if self.features == nil then
    self.features = {}
    self.feature_model_maps = {}
    self.feature_master_model_map = {}
  end
  assert(feat ~= nil, 'features can\'t be NULL!')
  self.features[name] = feat
  printf('feature [%s] added!\n', name)
  -- get model maps
  local model_maps = {}
  local master_model_map
  if name == 'EncDecA' then
    master_model_map = BModel.get_module_map({feat.enc_lstm_master, feat.dec_lstm_master})
  elseif name == 'NeuLexTransSoft' then
    master_model_map = BModel.get_module_map({feat.enc_lstm_master, feat.dec_softmax_master})
  else
    error(string.format('[%s] not supported yet!', name))
  end
  
  -- extract model maps
  for t = 1, feat.opts.seqLen do
    if name == 'EncDecA' then
      model_maps[t] = BModel.get_module_map({feat.enc_lstms[t], feat.dec_lstms[t]})
    elseif name == 'NeuLexTransSoft' then
      model_maps[t] = BModel.get_module_map({feat.enc_lstms[t], feat.dec_softmaxs[t]})
    else
      error(string.format('[%s] not supported yet!', name))
    end
    
    if t == 1 then
      print( table.keys( model_maps[t] ) )
    end
  end
  
  self.feature_model_maps[name] = model_maps
  self.feature_master_model_map[name] = master_model_map
end


function NeuDynFeat:generateData(x, x_mask, y, shuffle)
  self.features.EncDecA:validBatch(x, x_mask, y)
  local x_mask_t = x_mask:t()
  local Tx, Ty, bs = x:size(1), y:size(1), x:size(2)
  local align = torch.FloatTensor(Ty-1, bs, Tx)
  for t = 1, Ty - 1 do
    align[{ t, {}, {} }] = self.feature_model_maps.EncDecA[t].attention.output:float()
    align[{ t, {}, {} }]:cmul( x_mask_t )
  end
  self.features.NeuLexTransSoft:validBatch(x, x_mask, align, y)
  
  -- get model hidden states h_last and outputs log p(y|x)
  local data = {}
  local rndIdx
  if shuffle then
    rndIdx = torch.randperm(Ty-1)
  else
    rndIdx = torch.linspace(1, Ty-1, Ty-1)
  end
  
  for i = 1, Ty - 1 do
    local t = rndIdx[i]
    if t >= self.features.NeuLexTransSoft.opts.decStart then
      local h1, y1 = self.feature_model_maps.EncDecA[t].h_last.output:clone(), self.feature_model_maps.EncDecA[t].y_softmax.output:clone()
      local h2, y2 = self.feature_model_maps.NeuLexTransSoft[t].h_last.output:clone(), self.feature_model_maps.NeuLexTransSoft[t].y_softmax.output:clone()
      
      data[#data + 1] = {{h1, y1, h2, y2}, t}
    end
  end
  
  return data
end


function NeuDynFeat:trainBatch_local(data, i, y, sgdParam)
  
  local function feval(params_)
    if self.params ~= params_ then
      self.params:copy(params_)
    end
    self.grads:zero()
    
    local input, t = unpack(data[i])
    local y_pred = self.soft_gate:forward(input)
    local loss = self.criterion:forward({y_pred, y[{ t+1, {} }], self.opts.batchSize})
    local df_pred = self.criterion:backward({y_pred, y[{ t+1, {} }], self.opts.batchSize})
    self.soft_gate:backward(input, df_pred)
    
    return loss, self.grads
  end
  
  local _, loss_ = self.optimMethod(feval, self.params, sgdParam)
  return loss_[1]
end


function NeuDynFeat:validBatch_local(data, i, y)
  local input, t = unpack(data[i])
  local y_pred = self.soft_gate:forward(input)
  local loss = self.criterion:forward({y_pred, y[{ t+1, {} }], self.opts.batchSize})
  
  return loss, y_pred
end


function NeuDynFeat:trainBatch(x, x_mask, y, sgdParam)
  local data = self:generateData(x, x_mask, y, true)
  self.soft_gate:training()
  
  local loss = 0
  for i = 1, #data do
    local loss_ = self:trainBatch_local(data, i, y, sgdParam)
    loss = loss + loss_
  end
  
  return loss
end


function NeuDynFeat:validBatch(x, x_mask, y)
  local data = self:generateData(x, x_mask, y, false)
  self.soft_gate:evaluate()
  
  local loss = 0
  for i = 1, #data do
    local loss_ = self:validBatch_local(data, i, y)
    loss = loss + loss_
  end
  
  return loss
end


function NeuDynFeat:evaluateMode()
  self.soft_gate:evaluate()
end


function NeuDynFeat:fprop()
  local h1, y1 = self.feature_master_model_map.EncDecA.h_last.output:clone(), self.feature_master_model_map.EncDecA.y_softmax.output:clone()
  local h2, y2 = self.feature_master_model_map.NeuLexTransSoft.h_last.output:clone(), self.feature_master_model_map.NeuLexTransSoft.y_softmax.output:clone()
  
  local y_pred = self.soft_gate:forward({h1, y1, h2, y2})
  
  return y_pred
end



