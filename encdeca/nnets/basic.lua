
require 'torch'

local BModel = torch.class('BModel')

function BModel:__init()
  self.name = 'Basic Model, name needed!'
end

function BModel:save(modelPath, saveOpts)
  if not modelPath:ends('.t7') then
    modelPath = modelPath .. '.t7'
  end
  
  if self.params:type() == 'torch.CudaTensor' then
    torch.save(modelPath, self.params:float())
  else
    torch.save(modelPath, self.params)
  end
  
  if saveOpts then
    local optPath = modelPath:sub(1, -4) .. '.state.t7'
    torch.save(optPath, self.opts)
  end
end

function BModel:load(modelPath)
  local md = torch.load(modelPath)
  print('load model size = ' .. md:size(1))
  print('model size = ' .. self.params:size(1))
  self.params:copy( md )
end

function BModel:setModel(params)
  self.params:copy(params)
end

function BModel:getModel(outModel)
  return outModel:copy(self.params)
end

function BModel:print(msg)
  if msg == nil then
    xprint('the model is [%s]\n', self.name)
  else
    xprintln('[%s] %s', self.name, msg)
  end
end

function BModel.get_module_map(mods)
  local mdict = {}
  
  local function get_map(m)
    for _, node in ipairs(m.forwardnodes) do
      if node.data.annotations.name then
        mdict[node.data.annotations.name] = node.data.module
      end
    end
  end
  
  if torch.type(mods) == 'table' then
    for _, mod in ipairs(mods) do
      get_map(mod)
    end
  else
    get_map(mods)
  end
  
  return mdict
end


