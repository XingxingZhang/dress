
require '.'
require 'shortcut'

local PostPro = torch.class('PostProcessor')

function PostPro.recoverNER(src_file, dst_file, map_file, out_dst_file)
  local map_all = torch.load(map_file)
  local map
  if src_file:ends('.valid.src') then
    map = map_all.valid
  elseif src_file:ends('.test.src') then
    map = map_all.test
  end
  assert( xcountlines(src_file) == xcountlines(dst_file), 'should have the same number of lines' )
  
  local fin_src = io.open(src_file, 'r')
  local fin_dst = io.open(dst_file, 'r')
  local fout_dst = io.open(out_dst_file, 'w')
  local cnt = 0
  local ne_found_cnt = 0
  while true do
    local src_line = fin_src:read()
    local dst_line = fin_dst:read()
    if src_line == nil then
      assert(dst_line == nil)
      break
    end
    cnt = cnt + 1
    local ne_map = map[cnt]
    
    local swords = src_line:splitc(' \t')
    local dwords = dst_line:splitc(' \t')
    local snes = {}
    for _, wd in ipairs(swords) do
      if ne_map[wd] then snes[wd] = true end
    end
    
    local match_cnt, tot_cnt = 0, 0
    for i, wd in ipairs(dwords) do
      if xmatches(wd, '%u+@%d+') then
        tot_cnt = tot_cnt + 1
        if snes[wd] then
          match_cnt = match_cnt + 1
          dwords[i] = ne_map[wd]
        end
      end
    end
    
    fout_dst:write(table.concat(dwords, ' ') .. '\n')
    
    if match_cnt == tot_cnt then
      ne_found_cnt = ne_found_cnt + 1
    else
      print('src line = ', src_line)
      print('dst line = ', dst_line)
      print('ne map = ', ne_map)
      
      print(snes)
      print('dst after replace NER = ', table.concat(dwords, ' '))
      print('========================')
      print '\n\n'
    end
    --[[
    if cnt == 3 then
      break
    end
    --]]
  end
  fin_src:close()
  fin_dst:close()
  fout_dst:close()
  
  printf('ne found rate %d / %d = %f\n', ne_found_cnt, cnt, ne_found_cnt/cnt)
end

function PostPro.compareOri(src_file, dst_file)
  local fin_src = io.open(src_file)
  local fin_dst = io.open(dst_file)
  local cnt, repeat_cnt = 0, 0
  local unk_cnt, unk_sent_cnt = 0 , 0
  while true do
    local src_line = fin_src:read()
    local dst_line = fin_dst:read()
    if src_line == nil then
      assert(dst_line == nil)
      break
    end
    local swords = src_line:trim():splitc(' ')
    local dwords = dst_line:trim():splitc(' ')
    
    local has_unk = false
    for _, wd in ipairs(dwords) do
      if wd == 'UNK' then
        unk_cnt = unk_cnt + 1
        has_unk = true
      end
    end
    if has_unk then
      unk_sent_cnt = unk_sent_cnt + 1
    end
    
    
    if #swords == #dwords then
      local same = true
      
      for i = 1, #swords do
        if swords[i] ~= dwords[i] and dwords[i] ~= 'UNK' then
          same = false
        end
      end
      if same then
        repeat_cnt = repeat_cnt + 1
      end
    end
    cnt = cnt + 1
  end
  fin_src:close()
  fin_dst:close()
  
  printf('repeat rate: %d / %d = %f\n', repeat_cnt, cnt, repeat_cnt / cnt)
  printf('unk count = %d, unk sent count = %d\n', unk_cnt, unk_sent_cnt)
end

local function main()
  --[[
  local ori_src_file = '/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/PWKP/an_ner/PWKP_108016.tag.80.aner.ori.test.src'
  local src_file = '/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/PWKP/an_ner/PWKP_108016.tag.80.aner.test.src'
  local ref_file = '/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/PWKP/an_ner/PWKP_108016.tag.80.aner.ori.test.dst'
  local dst_file = '/disk/scratch/XingxingZhang/encdec/sent_simple/encdec_attention_PWKP/sample_0.001.256.dot.2L.adam.test'
  local map_file = '/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/PWKP/an_ner/PWKP_108016.tag.80.aner.map.t7'
  local dst_out_file = 'test.out.txt'
  --]]
  
  local ori_src_file = '/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/PWKP/an_ner/PWKP_108016.tag.80.aner.ori.valid.src'
  local src_file = '/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/PWKP/an_ner/PWKP_108016.tag.80.aner.valid.src'
  local ref_file = '/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/PWKP/an_ner/PWKP_108016.tag.80.aner.ori.valid.dst'
  local dst_file = '/disk/scratch/XingxingZhang/encdec/sent_simple/encdec_attention_PWKP/sample_0.001.256.dot.2L.adam.valid'
  local map_file = '/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/PWKP/an_ner/PWKP_108016.tag.80.aner.map.t7'
  local dst_out_file = 'valid.out.txt'
  
  -- local dst_out_file = 'valid.out.txt'
  PostProcessor.recoverNER(src_file, dst_file, map_file, dst_out_file)
  local cmd = string.format('./scripts/multi-bleu.perl %s < %s', ref_file, dst_out_file)
  os.execute(cmd)
  
  PostProcessor.compareOri(ori_src_file, dst_out_file)
end

main()
