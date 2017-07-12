
local ModelOpts = {}

function ModelOpts.getOpts()
  local cmd = torch.CmdLine()
  cmd:text('====== Encoder-Decoder LSTM ======')
  cmd:option('--seed', 123, 'random seed')
  cmd:option('--model', 'EncDec', 'model options: currently only support EncDec and EncDecA')
  cmd:option('--attention', 'dot', 'attention type: dot or general')
  cmd:option('--train', '/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/PWKP/PWKP_108016.80.train.sort', 'train file')
  cmd:option('--freqCut', 1, 'for word frequencies')
  cmd:option('--ignoreCase', false, 'whether you will ignore the case')
  cmd:option('--valid', '/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/PWKP/PWKP_108016.80.valid.sort', 'valid file')
  cmd:option('--test', '/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/PWKP/PWKP_108016.80.test.sort', 'test file (in default: no test file)')
  cmd:option('--validout', 'valid.out', 'valid decode file')
  cmd:option('--testout', 'test.out', 'test decode file')
  cmd:option('--maxEpoch', 30, 'maximum number of epochs')
  cmd:option('--batchSize', 64, '')
  cmd:option('--validBatchSize', 16, '')
  cmd:option('--nin', 100, 'word embedding size')
  cmd:option('--nhid', 200, 'hidden unit size')
  cmd:option('--nlayers', 1, 'number of hidden layers')
  cmd:option('--lr', 0.1, 'learning rate')
  cmd:option('--lrDiv', 0, 'learning rate decay when there is no significant improvement. 0 means turn off')
  cmd:option('--minImprovement', 1.001, 'if improvement on log likelihood is smaller then patient --')
  cmd:option('--optimMethod', 'AdaGrad', 'optimization algorithm')
  cmd:option('--gradClip', 5, '> 0 means to do Pascanu et al.\'s grad norm rescale http://arxiv.org/pdf/1502.04623.pdf; < 0 means to truncate the gradient larger than gradClip; 0 means turn off gradient clip')
  cmd:option('--initRange', 0.1, 'init range')
  cmd:option('--initHidVal', 0.01, 'init values for hidden states')
  cmd:option('--seqLen', 82, 'maximum seqence length')
  cmd:option('--useGPU', false, 'use GPU')
  cmd:option('--patience', 1, 'stop training if no lower valid PPL is observed in [patience] consecutive epoch(s)')
  cmd:option('--save', 'model.t7', 'save model path')
  
  cmd:text()
  cmd:text('Options for NCE')
  cmd:option('--nneg', 20, 'number of negative samples')
  cmd:option('--power', 0.75, 'for power for unigram frequency')
  cmd:option('--lnZ', 9, 'default normalization term')
  cmd:option('--learnZ', false, 'learn the normalization constant Z')
  cmd:option('--normalizeUNK', false, 'if normalize UNK or not')
  
  cmd:text()
  cmd:text('Options for long jobs')
  cmd:option('--savePerEpoch', false, 'save model every epoch')
  cmd:option('--saveBeforeLrDiv', false, 'save model before lr div')
  
  cmd:text()
  cmd:text('Options for regularization')
  cmd:option('--dropout', 0, 'dropout rate (dropping)')
  
  cmd:text()
  cmd:text('Options for Word Embedding initialization')
  cmd:option('--wordEmbedding', '', 'word embedding path')
  cmd:option('--embedOption', 'init', 'options: init, fineTune (if you use fineTune option, you must specify fineTuneFactor)')
  cmd:option('--fineTuneFactor', 0, '0 mean not upates, other value means such as 0.01')
  
  local opts = cmd:parse(arg)
  ModelOpts.initOpts(opts)
  
  return opts
end

function ModelOpts.initOpts(opts)
  -- for different optimization algorithms
  local optimMethods = {'AdaGrad', 'Adam', 'AdaDelta', 'SGD'}
  if not table.contains(optimMethods, opts.optimMethod) then
    error('invalid optimization method! ' .. opts.optimMethod)
  end
  
  opts.curLR = opts.lr
  opts.minLR = 1e-7
  opts.sgdParam = {learningRate = opts.lr}
  if opts.optimMethod == 'AdaDelta' then
    opts.rho = 0.95
    opts.eps = 1e-6
    opts.sgdParam.rho = opts.rho
    opts.sgdParam.eps = opts.eps
  elseif opts.optimMethod == 'SGD' then
    if opts.lrDiv <= 1 then
      opts.lrDiv = 2
    end
  end
  
  torch.manualSeed(opts.seed)
  if opts.useGPU then
    require 'cutorch'
    require 'cunn'
    cutorch.manualSeed(opts.seed)
  end
end

return ModelOpts

