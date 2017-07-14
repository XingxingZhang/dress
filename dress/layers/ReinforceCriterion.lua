--
-- based on Facebook's ReinforceCriterion.lua
-- at https://github.com/facebookresearch/MIXER
--

local ReinforceCriterion, parent = torch.class('nn.ReinforceCriterion',
                                               'nn.Module')
-- This criterion implements the REINFORCE algorithm under the assumption that
-- the reward does not depend on the model parameters.
-- The constructor takes as input a function which is used to compute the reward
-- given the ground truth input sequence, the generated sequence and the current
-- time step.
-- The input to the criterion is a table whose entries are the output of the
-- RNN at a certain time step, namely:
-- (chosen_word, predicted_cumulative_reward)_t
-- It computes the total reward and bprop the derivative
-- w.r.t. the above provided inputs.
--  reward_func: user provided function to compute the reward
--   given ground truth, current sequence and current time step.
-- seq_length is the length of the sequence we use
-- skips is the number of time steps we skip from the input and target (init)
-- weight is the weight on the loss produced by this criterion
-- weight_predictive_reward is the weight on the gradient of the cumulative
--   reward predictor (only)
function ReinforceCriterion:__init(sampleStart, seqLen, batchSize)
   parent.__init(self)
   self.sampleStart = sampleStart
   self.cum_reward = torch.Tensor(seqLen, batchSize)
   self.grad_rf_sample = torch.Tensor(seqLen, batchSize)
   self.seqLen = seqLen
   self.grad_exp_reward = {}
   for t = 1, seqLen do
     self.grad_exp_reward[t] = torch.Tensor(batchSize, 1)
   end
   
   --[[
   self.gradInput = {}
   self.seq_length = seq_length
   for tt = 1, seq_length do
      self.gradInput[tt] = {}
      self.gradInput[tt][1] = torch.Tensor()
      self.gradInput[tt][2] = torch.Tensor()
   end
   self.sizeAverage = false
   self.reward_func = reward_func
   self.reward = torch.Tensor()
   self.cumreward = torch.Tensor()
   self.skips = (skips == nil) and 1 or skips
   assert(self.skips <= seq_length)
   assert(seq_length >= self.skips)
   -- by default, update the cumulative reward predictor
   -- at a slower pace.
   self.weight_predictive_reward =
       (weight_predictive_reward == nil) and 0.01 or weight_predictive_reward
   self.weight = (weight == nil) and 1 or weight
   self.num_samples = 0
   self.normalizing_coeff = 1
   self.eos = eos_index
   self.padding = padding_index
   self.reset = torch.Tensor()
   --]]
end

function ReinforceCriterion:setSampleStart(start)
  self.sampleStart = start
end

function ReinforceCriterion:type(tp)
   parent.type(self, tp)
   self.cum_reward:type(tp)
   self.grad_rf_sample:type(tp)
   for t = 1, self.seqLen do
     self.grad_exp_reward[t]:type(tp)
   end
   
   return self
end

function ReinforceCriterion:set_weight(ww)
   self.weight = ww
end

function ReinforceCriterion:setSampleStart(start)
  self.sampleStart = start
end

function ReinforceCriterion:updateOutput(input)
  local rf_sample, exp_reward, true_reward, reward_mask = unpack(input)
  local Ty_rf = rf_sample:size(1)
  self.cum_reward:resizeAs(true_reward):zero()
  for t = Ty_rf - 1, self.sampleStart, -1 do
    self.cum_reward[{ t, {} }]:add( true_reward[{ t+1, {} }], self.cum_reward[{ t+1, {} }] )
    self.cum_reward[{ t, {} }]:cmul( reward_mask[{ t+1, {} }] )
  end
  
  local seq_reward = self.cum_reward[{ self.sampleStart, {} }]
  local n_seq = seq_reward:ne(0):sum()
  
  return -seq_reward:sum() / rf_sample:size(2), n_seq
end


--[[
-- input is a table storing the tuple:
-- (chosen_word, predicted_cumulative_reward)_t, t=1..T
-- target is also a table storing the labels at each time step.
function ReinforceCriterion:updateOutput(input, target)
   -- compute the reward at each time step
   local mbsz = target[1]:size(1)
   local num_steps = self.seq_length - self.skips + 1
   self.reward:resize(mbsz, num_steps)
   self.cum_reward:resize(mbsz, num_steps)
   self.num_samples = 0
   for tt = self.seq_length, self.skips, -1 do
      local shifted_tt = tt - self.skips + 1
      self.reward:select(2, shifted_tt):copy(
         self.reward_func:get_reward(target, input, tt))
      if tt == self.seq_length then
         self.cumreward:select(2, shifted_tt):copy(
            self.reward:select(2, shifted_tt))
      else
         self.cumreward:select(2, shifted_tt):add(
            self.cumreward:select(2, shifted_tt + 1),
            self.reward:select(2, shifted_tt))
      end
   end
   self.num_samples = self.reward_func:num_samples(target, input)
   self.normalizing_coeff =
      self.weight / (self.sizeAverage and self.num_samples or 1)
   -- here there is a "-" because we minimize
   self.output = - self.cumreward:select(2,1):sum() * self.normalizing_coeff
   return self.output, self.num_samples
end
--]]


function ReinforceCriterion:updateGradInput(input)
  local rf_sample, exp_reward, true_reward, reward_mask = unpack(input)
  local Ty_rf = rf_sample:size(1)
  self.grad_rf_sample:resizeAs(rf_sample):zero()
  local grad_exp_reward = {}
  for t = Ty_rf - 1, self.sampleStart, -1 do
    self.grad_rf_sample[{ t+1, {} }] = (exp_reward[t]:squeeze() - self.cum_reward[{ t, {} }])
    self.grad_rf_sample[{ t+1, {} }]:div(rf_sample:size(2))
    self.grad_rf_sample[{ t+1, {} }]:cmul( reward_mask[{ t+1, {} }] )
    self.grad_exp_reward[t]:resizeAs(exp_reward[t]):zero()
    self.grad_exp_reward[t]:copy( self.grad_rf_sample[{ t+1, {} }] )
    grad_exp_reward[t] = self.grad_exp_reward[t]
  end
  
  for t = 1, self.sampleStart - 1 do
    self.grad_exp_reward[t]:zero()
    grad_exp_reward[t] = self.grad_exp_reward[t]
  end
  
  self.gradInput = {self.grad_rf_sample, grad_exp_reward}
  
  return self.gradInput
end


--[[
-- bprop through input at each time step.
-- derivative through chosen action is:
-- (predicted_cumulative_reward - actual_cumulative_reward)_t.
function ReinforceCriterion:updateGradInput(input, target)
   local mbsz = target[1]:size(1)
   for tt = self.seq_length, self.skips, -1 do
      local shifted_tt = tt - self.skips + 1
      -- derivative w.r.t. chosen action
      self.gradInput[tt][1]:resizeAs(input[tt][1])
      self.gradInput[tt][1]:add(
         input[tt][2]:squeeze(), -1, self.cumreward:select(2, shifted_tt))
      self.gradInput[tt][1]:mul(self.normalizing_coeff)
      -- reset gradient to 0 if input (at any time) has PAD
      self.reset:resize(mbsz)
      self.reset:ne(input[tt][1], self.padding) -- set in RNNreinforce
      self.gradInput[tt][1]:cmul(self.reset)
      -- copy over to the other input gradient as well
      self.gradInput[tt][2]:resizeAs(input[tt][2])
      self.gradInput[tt][2]:copy(self.gradInput[tt][1])
      self.gradInput[tt][2]:mul(self.weight_predictive_reward)
   end
   -- fill the remaining (skipped) steps with 0s
   for tt = self.skips - 1, 1, -1 do
      self.gradInput[tt][1]:resizeAs(input[tt][1])
      self.gradInput[tt][1]:fill(0)
      self.gradInput[tt][2]:resizeAs(input[tt][2])
      self.gradInput[tt][2]:fill(0)
   end
   return self.gradInput
end
--]]


function ReinforceCriterion:get_num_samples(input, target)
   return self.reward_func:num_samples(target, input)
end

function ReinforceCriterion:reset_reward()
    return self.reward_func:reset_vars()
end

function ReinforceCriterion:get_corpus_score()
    return self.reward_func:get_corpus_score()
end

function ReinforceCriterion:get_counts_corpus(target, pred)
    return self.reward_func:get_counts_corpus(target, pred)
end

function ReinforceCriterion:training_mode()
    self.reward_func:training_mode()
end

function ReinforceCriterion:test_mode()
    self.reward_func:test_mode()
end
