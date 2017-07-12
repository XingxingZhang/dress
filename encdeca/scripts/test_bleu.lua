
local cmd = string.format('./multi-bleu.perl %s < %s', 'ref.txt', 'hop.txt')
-- os.execute(cmd)
local cmdout = io.popen(cmd)
for line in cmdout:lines() do
  print(line)
end


