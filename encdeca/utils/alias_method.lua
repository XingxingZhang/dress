
local AliasMethod = torch.class('AliasMethod')

function AliasMethod:__init(probs)
  local function initArray(N, val)
    local arr = {}
    for i = 1, N do
      arr[i] = val
    end
    return arr
  end
  
  local N = #probs
  local probTable = initArray(N, 0)
  local aliasTable = initArray(N, 1)
  
  local smaller, larger = {}, {}
  for i, p in ipairs(probs) do
    probTable[i] = N * p
    if probTable[i] < 1.0 then
      smaller[#smaller + 1] = i
    else
      larger[#larger + 1] = i
    end
  end
  
  local smallerSize, largerSize = #smaller, #larger
  
  while smallerSize > 0 and largerSize > 0 do
    local small = smaller[smallerSize]
    smallerSize = smallerSize - 1
    local large = larger[largerSize]
    largerSize = largerSize - 1
    
    aliasTable[small] = large
    probTable[large] = probTable[large] - (1.0 - probTable[small])
    if probTable[large] < 1.0 then
      smallerSize = smallerSize + 1
      smaller[smallerSize] = large
    else
      largerSize = largerSize + 1
      larger[largerSize + 1] = large
    end
  end
  
  self.probTable = torch.DoubleTensor(probTable)
  self.aliasTable = torch.LongTensor(aliasTable)
  self.size = N
end

function AliasMethod:drawBatch(N)
  local rndIdxs = (torch.DoubleTensor(N):uniform(0, 1) * self.size + 1):long()
  local probs = self.probTable:index(1, rndIdxs)
  local coins = torch.DoubleTensor(N):uniform(0, 1)
  local rndOut = torch.LongTensor(N)
  local bl = torch.lt(coins, probs)
  rndOut[bl] = rndIdxs[bl]
  local nbl = (-bl + 1)
  rndOut[nbl] = self.aliasTable:index(1, rndIdxs[nbl])
  
  return rndOut
end

