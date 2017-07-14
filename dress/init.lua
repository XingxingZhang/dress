
function importdir(dpath)
  package.path = dpath .. "/?.lua;" .. package.path
end

-- load all local packages
for dir in paths.iterdirs(".") do
  -- print(dir)
  importdir(dir)
end

require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'shortcut'