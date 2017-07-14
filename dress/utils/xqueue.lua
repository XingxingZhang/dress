
local XQueue = torch.class('XQueue')

function XQueue:__init(maxn)
  self.MAXN = (maxn or 4294967294) + 1  -- 2^32 - 1, max unsigned int32 in other languages, should be big enough
  self.front = 0
  self.rear = 0
  self.Q = {}
end

function XQueue:push(v)
  local nextRear = self:nextPos(self.rear)
  if nextRear == self.front then error('queue is full!!!') end
  self.rear = nextRear
  self.Q[nextRear] = v
end

function XQueue:pop()
  self.front = self:nextPos(self.front)
  local rval = self.Q[self.front]
  self.Q[self.front] = nil
  return rval
end

function XQueue:top()
  return self.Q[self:nextPos(self.front)]
end

function XQueue:isEmpty()
  return self.front == self.rear
end

function XQueue:nextPos(x)
  local pos = x + 1
  return pos == self.MAXN and 0 or pos
end

function XQueue:isFull()
  return self:nextPos(self.rear) == self.front
end

function XQueue:printAll()
  if not self:isEmpty() then
    print '==queue elements=='
    local i = self:nextPos(self.front)
    while true do
      print(self.Q[i])
      if i == self.rear then break end
      i = self:nextPos(i)
    end
    print '==queue elements end=='
  end
end



