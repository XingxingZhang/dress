
--[[
-- when y contains zeros
--]]

local EMaskedClassNLLCriterion, parent = torch.class('EMaskedClassNLLCriterion', 'nn.Module')

function EMaskedClassNLLCriterion:__init()
  parent.__init(self)
end

function EMaskedClassNLLCriterion:updateOutput(input_)
  local input, target, div = unpack(input_)
  if input:dim() == 2 then
    local nll = 0
    local n = target:size(1)
    for i = 1, n do
      if target[i] ~= 0 then
        nll = nll - input[i][target[i]]
      end
    end
    self.output = nll / div
    return self.output
  else
    error('input must be matrix! Note only batch mode is supported!')
  end
end

function EMaskedClassNLLCriterion:updateGradInput(input_)
  local input, target, div = unpack(input_)
  -- print('self.gradInput', self.gradInput)
  self.gradInput:resizeAs(input)
  self.gradInput:zero()
  local er = -1 / div
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

