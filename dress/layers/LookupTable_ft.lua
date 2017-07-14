local LookupTable, parent = torch.class('LookupTable_ft', 'nn.LookupTable')

LookupTable.__version = 1

function LookupTable:__init(nIndex, nOutput, updateMask)
   parent.__init(self, nIndex, nOutput)
   self.updateMask = torch.ones(nIndex):view(nIndex, 1):expand(nIndex, nOutput)
   if updateMask ~= nil then
     self:setUpdateMask(updateMask)
   end
end

function LookupTable:setUpdateMask(updateMask)
  local nIndex = self.weight:size(1)
  local nOutput = self.weight:size(2)
  assert(updateMask:nElement() == nIndex)
  self.updateMask:copy( updateMask:view(nIndex, 1):expand(nIndex, nOutput) )
end

function LookupTable:applyGradMask()
  -- a slow solution
   self.gradWeight:cmul( self.updateMask )
end

-- we do not need to accumulate parameters when sharing
LookupTable.sharedAccUpdateGradParameters = LookupTable.accUpdateGradParameters

