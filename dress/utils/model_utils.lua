require 'torch'

local model_utils = {}

function model_utils.combine_all_parameters(...)
    --[[ like module:getParameters, but operates on many modules ]]--

    -- get parameters
    local networks = {...}
    local parameters = {}
    local gradParameters = {}
    for i = 1, #networks do
        local net_params, net_grads = networks[i]:parameters()

        if net_params then
            for _, p in pairs(net_params) do
                parameters[#parameters + 1] = p
            end
            for _, g in pairs(net_grads) do
                gradParameters[#gradParameters + 1] = g
            end
        end
    end
    
    
    local function storageInSet(set, storage)
        local storageAndOffset = set[torch.pointer(storage)]
        if storageAndOffset == nil then
            return nil
        end
        local _, offset = unpack(storageAndOffset)
        return offset
    end

    -- this function flattens arbitrary lists of parameters,
    -- even complex shared ones
    local function flatten(parameters)
        if not parameters or #parameters == 0 then
            return torch.Tensor()
        end
        local Tensor = parameters[1].new

        local storages = {}
        local nParameters = 0
        for k = 1,#parameters do
            local storage = parameters[k]:storage()
            if not storageInSet(storages, storage) then
                storages[torch.pointer(storage)] = {storage, nParameters}
                nParameters = nParameters + storage:size()
            end
        end

        local flatParameters = Tensor(nParameters):fill(1)
        local flatStorage = flatParameters:storage()

        for k = 1,#parameters do
            local storageOffset = storageInSet(storages, parameters[k]:storage())
            parameters[k]:set(flatStorage,
                storageOffset + parameters[k]:storageOffset(),
                parameters[k]:size(),
                parameters[k]:stride())
            parameters[k]:zero()
        end

        local maskParameters=  flatParameters:float():clone()
        local cumSumOfHoles = flatParameters:float():cumsum(1)
        local nUsedParameters = nParameters - cumSumOfHoles[#cumSumOfHoles]
        local flatUsedParameters = Tensor(nUsedParameters)
        local flatUsedStorage = flatUsedParameters:storage()

        for k = 1,#parameters do
            local offset = cumSumOfHoles[parameters[k]:storageOffset()]
            parameters[k]:set(flatUsedStorage,
                parameters[k]:storageOffset() - offset,
                parameters[k]:size(),
                parameters[k]:stride())
        end

        for _, storageAndOffset in pairs(storages) do
            local k, v = unpack(storageAndOffset)
            flatParameters[{{v+1,v+k:size()}}]:copy(Tensor():set(k))
        end

        if cumSumOfHoles:sum() == 0 then
            flatUsedParameters:copy(flatParameters)
        else
            local counter = 0
            for k = 1,flatParameters:nElement() do
                if maskParameters[k] == 0 then
                    counter = counter + 1
                    flatUsedParameters[counter] = flatParameters[counter+cumSumOfHoles[k]]
                end
            end
            assert (counter == nUsedParameters)
        end
        return flatUsedParameters
    end

    -- flatten parameters and gradients
    local flatParameters = flatten(parameters)
    local flatGradParameters = flatten(gradParameters)

    -- return new flat vector that contains all discrete parameters
    return flatParameters, flatGradParameters
end

function model_utils.combine_treelstm_parameters(emb, lstms, softmax)
    --This is a method only works for treelstm--

    -- get parameters. Note LSTMs will ignore the lookup table
    local parameters = {}
    local gradParameters = {}
    
    local function getAllParameters(net, parameters, gradParameters)
      local net_params, net_grads = net:parameters()
      if net_params then
        for _, p in pairs(net_params) do
              parameters[#parameters + 1] = p
          end
          for _, g in pairs(net_grads) do
              gradParameters[#gradParameters + 1] = g
          end
      end
    end
    
    -- get LSTM parameters. IGNORE LookupTable
    local function getLSTMParameters(lstms, parameters, gradParameters)
      for _, lstm in ipairs(lstms) do
        for _, node in ipairs(lstm.forwardnodes) do
          -- check IF this is a module and the module is not a lookup table
          if node.data.module and torch.type(node.data.module) ~= 'nn.LookupTable' then
             local mp,mgp = node.data.module:parameters()
             if mp and mgp then
                for i = 1,#mp do
                   table.insert(parameters, mp[i])
                   table.insert(gradParameters, mgp[i])
                end
             end
          end
        end
      end
    end
    
    getAllParameters(emb, parameters, gradParameters)
    getLSTMParameters(lstms, parameters, gradParameters)
    getAllParameters(softmax, parameters, gradParameters)
    
    local function storageInSet(set, storage)
        local storageAndOffset = set[torch.pointer(storage)]
        if storageAndOffset == nil then
            return nil
        end
        local _, offset = unpack(storageAndOffset)
        return offset
    end

    -- this function flattens arbitrary lists of parameters,
    -- even complex shared ones
    local function flatten(parameters)
        if not parameters or #parameters == 0 then
            return torch.Tensor()
        end
        local Tensor = parameters[1].new

        local storages = {}
        local nParameters = 0
        for k = 1,#parameters do
            local storage = parameters[k]:storage()
            if not storageInSet(storages, storage) then
                storages[torch.pointer(storage)] = {storage, nParameters}
                nParameters = nParameters + storage:size()
            end
        end

        local flatParameters = Tensor(nParameters):fill(1)
        local flatStorage = flatParameters:storage()

        for k = 1,#parameters do
            local storageOffset = storageInSet(storages, parameters[k]:storage())
            parameters[k]:set(flatStorage,
                storageOffset + parameters[k]:storageOffset(),
                parameters[k]:size(),
                parameters[k]:stride())
            parameters[k]:zero()
        end

        local maskParameters=  flatParameters:float():clone()
        local cumSumOfHoles = flatParameters:float():cumsum(1)
        local nUsedParameters = nParameters - cumSumOfHoles[#cumSumOfHoles]
        local flatUsedParameters = Tensor(nUsedParameters)
        local flatUsedStorage = flatUsedParameters:storage()

        for k = 1,#parameters do
            local offset = cumSumOfHoles[parameters[k]:storageOffset()]
            parameters[k]:set(flatUsedStorage,
                parameters[k]:storageOffset() - offset,
                parameters[k]:size(),
                parameters[k]:stride())
        end

        for _, storageAndOffset in pairs(storages) do
            local k, v = unpack(storageAndOffset)
            flatParameters[{{v+1,v+k:size()}}]:copy(Tensor():set(k))
        end

        if cumSumOfHoles:sum() == 0 then
            flatUsedParameters:copy(flatParameters)
        else
            local counter = 0
            for k = 1,flatParameters:nElement() do
                if maskParameters[k] == 0 then
                    counter = counter + 1
                    flatUsedParameters[counter] = flatParameters[counter+cumSumOfHoles[k]]
                end
            end
            assert (counter == nUsedParameters)
        end
        return flatUsedParameters
    end

    -- flatten parameters and gradients
    local flatParameters = flatten(parameters)
    local flatGradParameters = flatten(gradParameters)

    -- return new flat vector that contains all discrete parameters
    return flatParameters, flatGradParameters
end

function model_utils.share_lstms_lookuptable(emb, lstms)
  assert(torch.type(emb) == 'nn.LookupTable', 'emb MUST be a LookupTable')
  local embParams, embGradParams = emb:parameters()
  
  for _, lstm in ipairs(lstms) do
    for _, node in ipairs(lstm.forwardnodes) do
      -- check IF this is a module and the module is not a lookup table
      if node.data.module and torch.type(node.data.module) == 'nn.LookupTable' then
        local params, gradParams = node.data.module:parameters()
        for i = 1, #params do
          params[i]:set(embParams[i])
          gradParams[i]:set(embGradParams[i])
        end
        
        collectgarbage()
      end
    end
  end
end

-- share parameters of NCE in nce with Linear in softmax
function model_utils.share_nce_softmax(nce, softmax)
  local nce_module
  for _, node in ipairs(nce.forwardnodes) do
    if node.data.module and torch.type(node.data.module) == 'NCE' then
      nce_module = node.data.module
      print('NCE module found!')
    end
  end
  
  local linear_module
  for _, node in ipairs(softmax.forwardnodes) do
    if node.data.module and torch.type(node.data.module) == 'nn.Linear' then
      linear_module = node.data.module
      print('Linear module found!')
    end
  end
  
  linear_module.weight:set(nce_module.weight)
  linear_module.bias:zero()
  linear_module.gradWeight = nil
  linear_module.gradBias = nil
  
  collectgarbage()
  
  return nce_module, linear_module
  
end

-- share parameters of Linear module in softmax1 with that in softmax2
function model_utils.share_linear(softmax1, softmax2)
  local linear1
  for _, node in ipairs(softmax1.forwardnodes) do
    if node.data.module and torch.type(node.data.module) == 'nn.Linear' then
      linear1 = node.data.module
      print('first Linear module found!')
    end
  end
  
  local linear2
  for _, node in ipairs(softmax2.forwardnodes) do
    if node.data.module and torch.type(node.data.module) == 'nn.Linear' then
      linear2 = node.data.module
      print('second Linear module found!')
    end
  end
  
  linear2.weight:set(linear1.weight)
  linear2.bias:set(linear1.bias)
  linear2.gradWeight = nil
  linear2.gradBias = nil
  
  collectgarbage()
  
  return linear1, linear2
end

function model_utils.clone_many_times(net, T)
    local clones = {}

    local params, gradParams
    if net.parameters then
        params, gradParams = net:parameters()
        if params == nil then
            params = {}
        end
    end

    local paramsNoGrad
    if net.parametersNoGrad then
        paramsNoGrad = net:parametersNoGrad()
    end

    local mem = torch.MemoryFile("w"):binary()
    mem:writeObject(net)

    for t = 1, T do
        -- We need to use a new reader for each clone.
        -- We don't want to use the pointers to already read objects.
        local reader = torch.MemoryFile(mem:storage(), "r"):binary()
        local clone = reader:readObject()
        reader:close()

        if net.parameters then
            local cloneParams, cloneGradParams = clone:parameters()
            local cloneParamsNoGrad
            for i = 1, #params do
                cloneParams[i]:set(params[i])
                cloneGradParams[i]:set(gradParams[i])
            end
            if paramsNoGrad then
                cloneParamsNoGrad = clone:parametersNoGrad()
                for i =1,#paramsNoGrad do
                    cloneParamsNoGrad[i]:set(paramsNoGrad[i])
                end
            end
        end

        clones[t] = clone
        collectgarbage()
    end

    mem:close()
    return clones
end

function model_utils.disable_dropout(node)
  if type(node) == "table" and node.__typename == nil then
    for i = 1, #node do
      node[i]:apply(model_utils.disable_dropout)
    end
    return
  end
  if string.match(node.__typename, "Dropout") then
    node.train = false
    -- print 'found dropout; disable'
  end
end

function model_utils.enable_dropout(node)
  if type(node) == "table" and node.__typename == nil then
    for i = 1, #node do
      node[i]:apply(model_utils.enable_dropout)
    end
    return
  end
  if string.match(node.__typename, "Dropout") then
    node.train = true
    -- print 'found dropout; enable'
  end
end

function model_utils.load_embedding(emb, vocabPath, embedPath)
  require 'wordembedding'
  local vocab = torch.load(vocabPath)
  local wordEmbed = WordEmbedding(embedPath)
  wordEmbed:initMat(emb.weight, vocab)
  wordEmbed:releaseMemory()
  vocab = nil
  wordEmbed = nil
  collectgarbage()
end

function model_utils.clone_many_times_emb_ft(net, T)
    local clones = {}

    local params, gradParams
    if net.parameters then
        params, gradParams = net:parameters()
        if params == nil then
            params = {}
        end
    end

    local paramsNoGrad
    if net.parametersNoGrad then
        paramsNoGrad = net:parametersNoGrad()
    end

    local mem = torch.MemoryFile("w"):binary()
    mem:writeObject(net)
    
    local master_map = BModel.get_module_map(net)
    local lt_names = {}
    for k, v in pairs(master_map) do
      if k:find('lookup') ~= nil then
        table.insert(lt_names, k)
      end
    end
    
    for t = 1, T do
        -- We need to use a new reader for each clone.
        -- We don't want to use the pointers to already read objects.
        local reader = torch.MemoryFile(mem:storage(), "r"):binary()
        local clone = reader:readObject()
        reader:close()

        if net.parameters then
            local cloneParams, cloneGradParams = clone:parameters()
            local cloneParamsNoGrad
            for i = 1, #params do
                cloneParams[i]:set(params[i])
                cloneGradParams[i]:set(gradParams[i])
            end
            if paramsNoGrad then
                cloneParamsNoGrad = clone:parametersNoGrad()
                for i =1,#paramsNoGrad do
                    cloneParamsNoGrad[i]:set(paramsNoGrad[i])
                end
            end
        end
        
        local clone_map = BModel.get_module_map(clone)
        for _, k in ipairs(lt_names) do
          if master_map[k].updateMask ~= nil then
            clone_map[k].updateMask:set( master_map[k].updateMask )
          end
        end

        clones[t] = clone
        collectgarbage()
    end

    mem:close()
    return clones
end

function model_utils.load_embedding_init(emb, vocab, embedPath)
  require 'wordembedding'
  local wordEmbed = WordEmbedding(embedPath)
  wordEmbed:initMat(emb.weight, vocab)
  wordEmbed:releaseMemory()
  vocab = nil
  wordEmbed = nil
  collectgarbage()
end

function model_utils.load_embedding_fine_tune(emb, vocab, embedPath, ftFactor)
  require 'wordembedding_ft'
  local wordEmbed = WordEmbeddingFT(embedPath)
  local mask = wordEmbed:initMatFT(emb.weight, vocab, ftFactor)
  emb:setUpdateMask(mask)
  wordEmbed:releaseMemory()
  vocab = nil
  wordEmbed = nil
  collectgarbage()
end


function model_utils.show_nngraph_names(nngraph)
  -- nnNode:graphNodeName
  for _, node in ipairs(nngraph.forwardnodes) do
    print(node:graphNodeName())
  end
  
end

function model_utils.copy_table(to, from)
  assert(#to == #from)
  for i = 1, #to do
    to[i]:copy(from[i])
  end
end

return model_utils
