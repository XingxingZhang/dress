

local AnalyzeResults = {}

local function getlines(infile)
  local lines = {}
  local fin = io.open(infile)
  while true do
    local line = fin:read()
    if line == nil then break end
    table.insert(lines, line)
  end
  fin:close()
  
  return lines
end

function AnalyzeResults.analyze(src_file, ref_file, tst_file, out_file)
  local src_lines = getlines(src_file)
  local ref_lines = getlines(ref_file)
  local tst_lines = getlines(tst_file)
  
  local fout = io.open(out_file, 'w')
  for i = 1, #src_lines do
    fout:write('[source   ] = ' .. src_lines[i] .. '\n')
    fout:write('[reference] = ' .. ref_lines[i] .. '\n')
    fout:write('[target   ] = ' .. tst_lines[i] .. '\n')
    fout:write('\n\n')
  end
  fout:close()
end

local function main()
  local cmd = torch.CmdLine()
  cmd:option('--src', '/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/PWKP/an_ner/PWKP_108016.tag.80.aner.ori.test.src', 'src file')
	cmd:option('--ref', '/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/PWKP/an_ner/PWKP_108016.tag.80.aner.ori.test.dst', 'ref file')
	cmd:option('--tst', 'test.out.txt', 'tst file')
  cmd:option('--out', 'test.out.ana.txt', 'out file')
  local opts = cmd:parse(arg)
  AnalyzeResults.analyze(opts.src, opts.ref, opts.tst, opts.out)
end

if not package.loaded['analyze_results'] then
	main()
else
	return AnalyzeResults
end

