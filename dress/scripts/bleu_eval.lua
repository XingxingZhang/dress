
local BleuEval = {}

require 'pl'
stringx.import()

local function printf(s, ...)
	return io.write(s:format(...))
end

local function println(s, ...)
	printf(s .. '\n', ...)
end

local function htmlize(s)
  s = s:gsub('&', '&amp;')
  s = s:gsub('<', '&lt;')
  s = s:gsub('>', '&gt;')
  s = s:gsub('"', '&quot;')
  s = s:gsub('\'', '&apos;')
  
  return s
end

local function txt2xml(infile, outfile, setlabel)
	local template = [[
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE mteval SYSTEM "ftp://jaguar.ncsl.nist.gov/mt/resources/mteval-xml-v1.3.dtd">
<mteval>
<SET_LABEL setid="example_set" srclang="Arabic" trglang="English" refid="ref1" sysid="sample_system">
<doc docid="doc1" genre="nw">
CONTENT
</doc>
</SET_LABEL>
</mteval>
	]]
	local lines = {}
	local fin = io.open(infile)
	while true do 
		local line = fin:read()
		if line == nil then break end
    line = htmlize(line)
		table.insert(lines, line)
	end
	local texts = {}
	for i = 1, #lines do 
		table.insert(texts, '<p>')
		table.insert(texts, string.format('<seg id="%d"> %s </seg>', i, lines[i]))
		table.insert(texts, '</p>')
	end
	fin:close()
	local out_text = template:replace('SET_LABEL', setlabel)
	out_text = out_text:replace('CONTENT', table.concat(texts, '\n'))
	local fout = io.open(outfile, 'w')
	fout:write(out_text .. '\n')
	fout:close()
end

function BleuEval.eval(src_file, ref_file, tst_file, additional_opts)
  additional_opts = additional_opts or ''
	local src_xml = '__tmp__.src.xml'
	local ref_xml = '__tmp__.ref.xml'
	local tst_xml = '__tmp__.tst.xml'
	txt2xml(src_file, src_xml, 'srcset')
	txt2xml(ref_file, ref_xml, 'refset')
	txt2xml(tst_file, tst_xml, 'tstset')
  local eval_script_name = 'mteval-v13a.pl'
  if paths.dirp('./scripts') then
    eval_script_name = './scripts/' .. eval_script_name
  end
	local cmd = string.format('perl %s %s -s %s -r %s -t %s', eval_script_name, additional_opts, src_xml, ref_xml, tst_xml)
	local fin = io.popen(cmd, 'r')
	local s = fin:read('*a')
	print(s)
	local reg = 'BLEU score = ([^%s]+)'
	local istart, iend, bleu = s:find(reg)
	-- print(s:sub(istart, iend))
	print(bleu)

	os.remove(src_xml)
	os.remove(ref_xml)
	os.remove(tst_xml)
  
  return tonumber(bleu)
end

local function main()
	local cmd = torch.CmdLine()
	cmd:option('--src', '/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/PWKP/an_ner/PWKP_108016.tag.80.aner.ori.test.src', 'src file')
	cmd:option('--ref', '/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/PWKP/an_ner/PWKP_108016.tag.80.aner.ori.test.dst', 'ref file')
	cmd:option('--tst', 'test.out.txt', 'tst file')
  cmd:option('--opt', '', 'addtional opts for mteval-v13a')
	local opt = cmd:parse(arg)
	BleuEval.eval(opt.src, opt.ref, opt.tst, opt.opt)
end

if not package.loaded['bleu_eval'] then
	main()
else
	return BleuEval
end


