
-- normalizing constant is learned automatically
-- support learning Z automatically
-- I want to make it faster

local NCE, parent = torch.class('NCE', 'nn.LookupTable')

function NCE:__init(inputSize, outputSize, Z, learnZ)
   parent.__init(self, outputSize, inputSize)
   
   self.learnZ = learnZ
   if learnZ then
     self.bias = torch.Tensor(1)
     self.gradBias = torch.Tensor(1)
     print('learning Z')
     print('self.bias is acutally Z')
   else
     self.Z = Z
     print('Z is a hyper-parameter')
   end
   
   print('NCE is from nn.LookupTable')
end

function NCE:updateOutput(input)
  -- hs: 2D, y: 1D, y_neg: 2D (x, n_neg), y_prob: 1D, y_neg_prob: 2D (x, n_neg)
  local hs, y, y_neg, y_prob, y_neg_prob = unpack(input)
  local Who = self.weight
  if hs:dim() == 2 then
    
    if self.learnZ then
      self.Z = self.bias[1]
    end
    
    -- compute non-normalized softmax for y
    self.We_out = Who:index(1, y)
    local We_out = self.We_out
    local pos_a = torch.cmul(We_out, hs):sum(2)
    local p_rnn_pos = pos_a:exp():div(self.Z)
    local k = y_neg:size(2)
    -- local y_prob_2d = y_prob:view(y_prob:size(1), 1)
    self.P_pos = torch.cdiv( p_rnn_pos, (p_rnn_pos + y_prob * k) )  -- P_pos shape (seqlen * bs, 1)
    local P_pos = self.P_pos
    local log_P_pos = torch.log(P_pos)
    
    -- compute non-normalized softmax for negative examples of y, y_neg
    local y_neg_ = y_neg:view(y_neg:size(1) * y_neg:size(2))
    local We_out_n_ = Who:index(1, y_neg_)
    local n_hid = Who:size(2)
    self.We_out_n = We_out_n_:view( y_neg:size(1), y_neg:size(2), n_hid )
    local We_out_n = self.We_out_n
    local neg_a = torch.cmul( We_out_n, hs:view(hs:size(1), 1, hs:size(2)):expand(hs:size(1), y_neg:size(2), hs:size(2)) ):sum(3)
    local p_rnn_neg = neg_a:exp():div(self.Z)
    local k_y_neg_prob = y_neg_prob * k
    self.P_neg = torch.cdiv( k_y_neg_prob, (p_rnn_neg + k_y_neg_prob) )
    local P_neg = self.P_neg
    local log_P_neg = torch.log(P_neg)
    
    self.output = log_P_pos + log_P_neg:sum(2)
    
    return self.output
  else
    error('input must be 2D matrix, currently only support batch mode')
  end
end

function NCE:updateGradInput(input, gradOutput)
  -- hs: 2D, y: 1D, y_neg: 2D (x, n_neg), y_prob: 1D, y_neg_prob: 2D (x, n_neg)
  local hs, y, y_neg, y_prob, y_neg_prob = unpack(input)
  
  -- gradOutput: is the scale of the gradients, gradOutput can contain 0s;
  -- that is to say gradOutput can also be served as mask; shape: (bs*seq, 1)
  
  if self.gradInput then
    -- I can't see why self.gradInput:zero() is useful
    local nElement = self.gradInput:nElement()
    self.gradInput:resizeAs(hs)
    if self.gradInput:nElement() ~= nElement then
       self.gradInput:zero()
    end
    
    if hs:dim() == 2 then
      -- gradients from the positive samples
      -- take mask (gradOutput) into account
      
      self.d_P_pos = torch.cmul( (-self.P_pos + 1), gradOutput )
      local d_P_pos = self.d_P_pos
      self.gradInput:cmul( self.We_out, d_P_pos:expand(self.P_pos:size(1), hs:size(2)) )
      
      -- gradients from the negative samples
      -- take (gradOutput) into account
      self.d_P_neg = torch.cmul( (self.P_neg - 1), gradOutput:expand(gradOutput:size(1), self.P_neg:size(2)) )
      local d_P_neg = self.d_P_neg
      local d_hs = self.We_out_n:cmul( 
          d_P_neg:view(d_P_neg:size(1), d_P_neg:size(2), 1):expand(d_P_neg:size(1), d_P_neg:size(2), hs:size(2)) 
        )
      self.gradInput:add(d_hs:sum(2))
      
      return {self.gradInput}
    else
      error('input must be 2D matrix, currently only support batch mode')
    end
  end
  
end

function NCE:accGradParameters(input, gradOutput)
  -- hs: 2D, y: 1D, y_neg: 2D (x, n_neg), y_prob: 1D, y_neg_prob: 2D (x, n_neg)
  local hs, y, y_neg, y_prob, y_neg_prob = unpack(input)
  
  self:backCompatibility()
  
  if hs:dim() == 2 then
    local d_P_pos = self.d_P_pos
    
    if self.learnZ then
      self.gradBias:add( (-d_P_pos / self.Z):sum() )
    end
    
    local gradWeight_pos = torch.cmul( hs, d_P_pos:expand(self.P_pos:size(1), hs:size(2)) )
    
    y = self:makeInputContiguous(y)
    y = self.copiedInput and self._input or y
    self.gradWeight.nn.LookupTable_accGradParameters(self, y, gradWeight_pos, 1)
    
    local d_P_neg = self.d_P_neg
    
    if self.learnZ then
      self.gradBias:add( (-d_P_neg / self.Z):sum() )
    end
    
    local gradWeight_neg = torch.cmul( 
      hs:view(hs:size(1), 1, hs:size(2)):expand(hs:size(1), y_neg:size(2), hs:size(2)),
      d_P_neg:view(d_P_neg:size(1), d_P_neg:size(2), 1):expand(d_P_neg:size(1), d_P_neg:size(2), hs:size(2))
    )
    
    y_neg = self:makeInputContiguous(y_neg)
    y_neg = self.copiedInput and self._input or y_neg
    self.gradWeight.nn.LookupTable_accGradParameters(self, y_neg:view(-1), gradWeight_neg, 1)
    
    self.d_P_pos = nil
    self.d_P_neg = nil
  else
    error('input must be 2D matrix, currently only support batch mode')
  end
  
end

-- we do not need to accumulate parameters when sharing
NCE.sharedAccUpdateGradParameters = NCE.accUpdateGradParameters

function NCE:__tostring__()
  return torch.type(self) ..
      string.format('(%d -> %d)', self.weight:size(2), self.weight:size(1))
end
