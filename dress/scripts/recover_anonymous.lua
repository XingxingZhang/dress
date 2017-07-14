
include '../utils/shortcut.lua'

local function insert_label(name, label)
  local pos = name:rfind('.')
  return name:sub(1, pos-1) .. label .. name:sub(pos)
end

local function recover(train_file, valid_file, test_file, map_file)
  local map = torch.load(map_file)
  local train_file_out = insert_label(train_file, '.ori')
  local valid_file_out = insert_label(valid_file, '.ori')
  local test_file_out = insert_label(test_file, '.ori')
  
  local function recover_one(train_file, train_file_out, map_train)
    local fin_src = io.open(train_file .. '.src', 'r')
    local fin_dst = io.open(train_file .. '.dst', 'r')
    local fout_src = io.open(train_file_out .. '.src', 'w')
    local fout_dst = io.open(train_file_out .. '.dst', 'w')
    
    local cnt = 0
    while true do
      local src = fin_src:read()
      local dst = fin_dst:read()
      cnt = cnt + 1
      if src == nil then
        assert(dst == nil, 'should be the same number of lines')
        break
      end
      local map = map_train[cnt]
      local src_words = src:trim():splitc(' \t\r\n')
      local src_words_out = {}
      for _, word in ipairs(src_words) do
        if map[word] then
          table.insert(src_words_out, map[word])
        else
          table.insert(src_words_out, word)
        end
      end
      local dst_words = dst:trim():splitc(' \t\r\n')
      local dst_words_out = {}
      for _, word in ipairs(dst_words) do
        if map[word] then
          table.insert(dst_words_out, map[word])
        else
          table.insert(dst_words_out, word)
        end
      end
      fout_src:write(table.concat(src_words_out, ' ') .. '\n')
      fout_dst:write(table.concat(dst_words_out, ' ') .. '\n')
    end
    
    fin_src:close()
    fin_dst:close()
    fout_src:close()
    fout_dst:close()
  end
  
  recover_one(train_file, train_file_out, map.train)
  recover_one(valid_file, valid_file_out, map.valid)
  recover_one(test_file, test_file_out, map.test)
end

local function main()
  --[[
  local cmd = torch.CmdLine()
  cmd:text('== convert taged text to dataset (.t7)==')
  cmd:option('--train', '/afs/inf.ed.ac.uk/user/s12/s1270921/Desktop/data/PWKP/data_std/PWKP_108016.tag.80.train', 'tagged training set')
  cmd:option('--valid', '/afs/inf.ed.ac.uk/user/s12/s1270921/Desktop/data/PWKP/data_std/PWKP_108016.tag.80.valid', 'tagged validation set')
  cmd:option('--test', '/afs/inf.ed.ac.uk/user/s12/s1270921/Desktop/data/PWKP/data_std/PWKP_108016.tag.80.test', 'tagged test set')
  cmd:option('--dataset', '/afs/inf.ed.ac.uk/user/s12/s1270921/Desktop/data/PWKP/data_std/PWKP_108016.tag.80.t7', 'resulting dataset')
  cmd:option('--totext', false, 'output text file instead of .t7 file')
  
  local opts = cmd:parse(arg)
  --]]
  --[[
  local train_file = '/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/PWKP/an_ner/PWKP_108016.tag.80.aner.train'
  local valid_file = '/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/PWKP/an_ner/PWKP_108016.tag.80.aner.valid'
  local test_file = '/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/PWKP/an_ner/PWKP_108016.tag.80.aner.test'
  local map_file = '/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/PWKP/an_ner/PWKP_108016.tag.80.aner.map.t7'
  --]]
  local cmd = torch.CmdLine()
  cmd:text('== convert aner files back to  text==')
  cmd:option('--train', '/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/PWKP/an_ner/PWKP_108016.tag.80.aner.train', '')
  cmd:option('--valid', '/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/PWKP/an_ner/PWKP_108016.tag.80.aner.valid', '')
  cmd:option('--test', '/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/PWKP/an_ner/PWKP_108016.tag.80.aner.test', '')
  cmd:option('--map', '/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/PWKP/an_ner/PWKP_108016.tag.80.aner.map.t7', '')
  local opts = cmd:parse(arg)
  
  local train_file = opts.train
  local valid_file = opts.valid
  local test_file = opts.test
  local map_file = opts.map
  recover(train_file, valid_file, test_file, map_file)
end

main()
