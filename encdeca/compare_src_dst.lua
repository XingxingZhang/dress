
require '.'
require 'shortcut'

local function compare(infile1, infile2)
  local fin1 = io.open(infile1, 'r')
  local fin2 = io.open(infile2, 'r')
  
  local same_count, cnt = 0, 0
  while true do
    local line1 = fin1:read()
    local line2 = fin2:read()
    if line1 == nil then
      assert(line2 == nil)
      break
    end
    if line1 == line2 then
      same_count = same_count + 1
    end
    cnt = cnt + 1
  end
  fin1:close()
  fin2:close()
  
  printf('%d / %d = %f\n', same_count, cnt, same_count/cnt)
end

local function main()
  printf('let"s do it!\n')
  compare('/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/PWKP/an_ner/PWKP_108016.tag.80.aner.ori.train.src', 
    '/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/PWKP/an_ner/PWKP_108016.tag.80.aner.ori.train.dst')
  
  compare('/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/PWKP/an_ner/PWKP_108016.tag.80.aner.train.src', 
    '/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/PWKP/an_ner/PWKP_108016.tag.80.aner.train.dst')
end

main()
