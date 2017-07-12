
--[[
-- when y contains zeros
--]]

local MaskedClassNLLCriterion, parent = torch.class('MaskedClassNLLCriterion', 'nn.Module')

function MaskedClassNLLCriterion:__init()
  parent.__init(self)
end

function MaskedClassNLLCriterion:updateOutput(input_)
  local input, target = unpack(input_)
  if input:dim() == 2 then
    local nll = 0
    local n = target:size(1)
    for i = 1, n do
      if target[i] ~= 0 then
        nll = nll - input[i][target[i]]
      end
    end
    self.output = nll / target:size(1)
    return self.output
  else
    error('input must be matrix! Note only batch mode is supported!')
  end
end

function MaskedClassNLLCriterion:updateGradInput(input_)
  local input, target = unpack(input_)
  -- print('self.gradInput', self.gradInput)
  self.gradInput:resizeAs(input)
  self.gradInput:zero()
  local er = -1 / target:size(1)
  if input:dim() == 2 then
    local n = target:size(1)
    for i = 1, n do
      if target[i] ~= 0 then
        self.gradInput[i][target[i]] = er
      end
    end
    return self.gradInput
  else
    error('input must be matrix! Note only batch mode is supported!')
  end
end

