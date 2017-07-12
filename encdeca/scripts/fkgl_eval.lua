
local FKGLEval = {}

function FKGLEval.eval(infile)
  local script_path = 'readability/getFKGL.py'
  if paths.dirp('./scripts') then
    script_path = './scripts/' .. script_path
  end
	local cmd = script_path .. ' ' .. infile
  local fin = io.popen(cmd, 'r')
	local s = fin:read('*a')
  print('fkgl = ' .. s)
  local fkgl = tonumber(s)
  
  return fkgl
end

local function main()
  local cmd = torch.CmdLine()
  cmd:option('--tst', 'test.out.txt', 'test file')
  local opt = cmd:parse(arg)
  FKGLEval.eval(opt.tst)
end

if not package.loaded['fkgl_eval'] then
	main()
else
	return FKGLEval
end

