
--[[
This is used to compute the loss function of NCE
--]]

local NCEMaskedLoss, parent = torch.class('NCEMaskedLoss', 'nn.Module')

function NCEMaskedLoss:__init()
  parent.__init(self)
end

function NCEMaskedLoss:updateOutput(input_)
  local input, mask, div = unpack(input_)
  self.output = -( input:view(-1) * mask:view(-1) ) / div
  
  return self.output
end

function NCEMaskedLoss:updateGradInput(input_)
  local input, mask, div = unpack(input_)
  
  --[[
  print('size of input')
  print(input:size())
  
  print('size of self.gradInput')
  print(self.gradInput:size())
  --]]
  
  self.gradInput:resizeAs(input)
  self.gradInput:zero()
  -- print(self.gradInput)
  
  local mask_
  if mask:dim() == 2 then
    mask_ = mask:view(mask:size(1) * mask:size(2), 1)
  elseif mask:dim() == 1 then
    mask_ = mask:view(mask:size(1), 1)
  else
    error('mask must be matrix or vector!')
  end
  
  --[[
  print('mask_ size')
  print(mask_:size())
  print(mask_:type())
  --]]
  
  self.gradInput:copy(-mask_ / div)
  
  --[[
  print('size gradInput')
  print(self.gradInput:size())
  print(self.gradInput[{ {-10, -1} }])
  --]]
  
  return {self.gradInput}
end

