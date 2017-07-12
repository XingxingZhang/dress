
include '../utils/shortcut.lua'

-- anomymize NERs (PER, LOC, ORG, MISC) and numbers

function table.find(t, key)
  for i, v in ipairs(t) do
    if v == key then return i end
  end
  return nil
end

local function loadData(infile)
  local data = {}
  local example = {}
  local fin = io.open(infile, 'r')
  while true do
    local line = fin:read()
    if line == nil then break end
    line = line:trim()
    if line:len() == 0 then
      assert(#example == 2, 'MUST be two examples')
      table.insert(data, example)
      example = {}
    else
      table.insert(example, line:splitc(' \t'))
    end
  end
  
  fin:close()
  
  return data
end

local function showLength(data)
  print 'begin'
  for _, d in ipairs(data) do
    print(#d[2])
  end
  print 'end'
end

--[[
local function divideNER(ner_str)
  local n = ner_str:len()
  local pos = {}
  for i = n, 1, -1 do
    if ner_str:sub(i, i) == '/' then
      table.insert(pos, i)
      if #pos == 2 then break end
    end
  end
  
  return ner_str:sub(1, pos[2] - 1), ner_str:sub(pos[2] + 1, pos[1] - 1), ner_str:sub(pos[1] + 1)
end
--]]

local function divideNER(ner_str)
  local n = ner_str:len()
  local pos = {}
  for i = n, 1, -1 do
    if ner_str:sub(i, i) == '/' then
      table.insert(pos, i)
      if #pos == 2 then break end
    end
  end
  -- assert(#pos == 1)
  
  return ner_str:sub(1, pos[1] - 1), 'POS', ner_str:sub(pos[1] + 1)
end

local function isNum(s)
  local n = s:len()
  local dcnt, tcnt = 0, 0
  for i = 1, n do
    tcnt = tcnt + 1
    if tonumber(s:sub(i,i)) ~= nil then
      dcnt = dcnt + 1
    end
  end
  
  if dcnt == tcnt and tcnt > 0  then
    return true
  elseif tcnt - dcnt <= 1 then
    return tcnt > 1
  else
    return (tcnt - dcnt) / tcnt < 0.25
  end
end

local function toNESent(insent)
  local outsent = {}
  
  local last_ner = 'O'
  local ne = {}
  for _, item in ipairs(insent) do
    local wd, pos, ner = divideNER(item)
    if ner ~= 'MISC' and ner ~= 'PERSON' and ner ~= 'LOCATION' and ner ~= 'ORGANIZATION' then
      ner = 'O'
    end
    
    if ner == 'O' then
      if #ne > 0 then
        table.insert(outsent, {table.concat(ne, ' '), last_ner})
      end
      ne = {}
      if isNum(wd) then
        table.insert(outsent, {wd, 'NUMBER'})
      else
        table.insert(outsent, wd)
      end
    else
      if ner == last_ner then
        table.insert(ne, wd)
      else
        if #ne > 0 then
          table.insert(outsent, {table.concat(ne, ' '), last_ner})
        end
        ne = {wd}
      end
    end
    last_ner = ner
  end
  
  if #ne > 0 then
    table.insert(outsent, {table.concat(ne, ' '), last_ner})
  end
  ne = {}
  
  
  return outsent
end

local function processSentPair(src, dst)
  -- process source
  local nesent_src = toNESent(src)
  local nesent_dst = toNESent(dst)
  
  local tag2ne = {}
  local ne2tag_id = {}
  local src_out = {}
  local dst_out = {}
  
  local function insert_ne(ne, tag)
    local tag_id = ne2tag_id[ne]
    local isnew = false
    if tag_id == nil then
      local nelist = tag2ne[tag]
      local id
      if nelist == nil then
        nelist = {ne}
        id = 1
        tag2ne[tag] = nelist
      else
        assert(table.find(nelist, ne) == nil, 'should not contain the ne')
        table.insert(nelist, ne)
        id = #nelist
      end
      tag_id = string.format('%s@%d', tag, id)
      ne2tag_id[ne] = tag_id
      isnew = true
    end
    
    return tag_id, isnew
  end
  
  for i, item in ipairs(nesent_src) do
    if type(item) == 'table' then
      local ne, tag = unpack(item)
      local tag_id, _ = insert_ne(ne, tag)
      table.insert(src_out, tag_id)
    else
      table.insert(src_out, item)
    end
  end
  
  local new_ne_cnt = 0
  for i, item in ipairs(nesent_dst) do
    if type(item) == 'table' then
      local ne, tag = unpack(item)
      local tag_id, isnew = insert_ne(ne, tag)
      table.insert(dst_out, tag_id)
      if isnew then new_ne_cnt = new_ne_cnt + 1 end
    else
      table.insert(dst_out, item)
    end
  end
  
  local tag_id2ne = {}
  for k, v in pairs(ne2tag_id) do tag_id2ne[v] = k end
  -- return table.concat(src_out, ' '), table.concat(src_out, ' '), tag_id2ne, new_ne_cnt
  return src_out, dst_out, tag_id2ne, new_ne_cnt
end

local function findNEDiff(ssent, dsent)
  local pat = '%u+@%d+'
  local sset = {}
  local set = {}
  for m in ssent:gmatch(pat) do
    sset[m] = true
    set[m] = true
  end
  local dset = {}
  for m in dsent:gmatch(pat) do
    dset[m] = true
    set[m] = true
  end
  local sne, dne, sdne = {}, {}, {}
  for k, _ in pairs(set) do
    if sset[k] and dset[k] then
      sdne[#sdne + 1] = k
    elseif sset[k] then
      sne[#sne + 1] = k
    else
      dne[#dne + 1] = k
    end
  end
  
  return sne, dne, sdne
end

local function anonymize_test(infile, outfile)
  local data = loadData(infile)
  -- print(#data)
  -- showLength(data)
  -- table.sort(data, function(a, b) return #a[2] < #b[2] end)
  -- print 'sort done!'
  -- showLength(data)
  
  local cnt = 0
  local new_cnt, new_cnt1, new_cnt2 = 0, 0, 0
  for _, p in ipairs(data) do
    cnt = cnt + 1
    if cnt % 1000 == 0 then
      print(cnt)
    end
    -- printf('cnt = %d\n', cnt)
    local src_sent, dst_sent, tag_id2ne, new_ne_cnt = processSentPair(p[1], p[2])
    if new_ne_cnt == 1 then
      print(src_sent)
      print(dst_sent)
      print(tag_id2ne)
      print(new_ne_cnt)
      local sne, dne, sdne = findNEDiff(src_sent, dst_sent)
      print('src only')
      print(sne)
      print('dst only')
      print(dne)
      print('src dst common')
      print(sdne)
    end
    
    if new_ne_cnt > 0 then new_cnt = new_cnt + 1 end
    if new_ne_cnt > 1 then new_cnt1 = new_cnt1 + 1 end
    if new_ne_cnt > 2 then new_cnt2 = new_cnt2 + 1 end
  end
  
  printf('%d / %d = %f has new nes\n', new_cnt, cnt, new_cnt/cnt)
  printf('%d / %d = %f has more than one new nes\n', new_cnt1, cnt, new_cnt1/cnt)
  printf('%d / %d = %f has more than one new nes\n', new_cnt2, cnt, new_cnt2/cnt)
  
  local mapfile = outfile .. '.map'
end

local function anonymize(infile)
  local data = loadData(infile)
  
  local anonymous_data = {}
  -- source target --
  local cnt = 0
  for _, p in ipairs(data) do
    cnt = cnt + 1
    if cnt % 1000 == 0 then
      print(cnt)
    end
    local src_sent, dst_sent, tag_id2ne, new_ne_cnt = processSentPair(p[1], p[2])
    table.insert(anonymous_data, {src_sent, dst_sent, tag_id2ne})
  end
  
  table.sort(anonymous_data, function(a, b) return #a[2] < #b[2] end)
  
  return anonymous_data
end

local function totext(split, infile, label)
  -- local fin = io.open(infile)
  local pos = infile:rfind('.')
  local outfile = infile:sub(1, pos-1) .. label .. infile:sub(pos)
  print(outfile)
  local fout_src = io.open(outfile .. '.src', 'w')
  local fout_trg = io.open(outfile .. '.dst', 'w')
  local map = {}
  for _, item in ipairs(split) do
    fout_src:write(table.concat(item[1], ' ') .. '\n')
    fout_trg:write(table.concat(item[2], ' ') .. '\n')
    table.insert(map, item[3])
  end
  
  fout_src:close()
  fout_trg:close()
  return map
end

local function main()
  local cmd = torch.CmdLine()
  cmd:text('== convert taged text to dataset (.t7)==')
  cmd:text('WARNING: bug fixed in toNESent(insent); previous version has bug!!!')
  cmd:option('--train', '/afs/inf.ed.ac.uk/user/s12/s1270921/Desktop/data/PWKP/data_std/PWKP_108016.tag.80.train', 'tagged training set')
  cmd:option('--valid', '/afs/inf.ed.ac.uk/user/s12/s1270921/Desktop/data/PWKP/data_std/PWKP_108016.tag.80.valid', 'tagged validation set')
  cmd:option('--test', '/afs/inf.ed.ac.uk/user/s12/s1270921/Desktop/data/PWKP/data_std/PWKP_108016.tag.80.test', 'tagged test set')
  cmd:option('--dataset', '/afs/inf.ed.ac.uk/user/s12/s1270921/Desktop/data/PWKP/data_std/PWKP_108016.tag.80.t7', 'resulting dataset')
  cmd:option('--totext', false, 'output text file instead of .t7 file')
  
  local opts = cmd:parse(arg)
  
  local dataset = {}
  local timer = torch.Timer()
  dataset.train = anonymize(opts.train)
  print 'create training done!'
  dataset.valid = anonymize(opts.valid)
  print 'create valiation done!'
  dataset.test = anonymize(opts.test)
  print 'create test done!'
  xprintln('time spend %fs', timer:time().real)
  timer:reset()
  if not opts.totext then
    torch.save(opts.dataset, dataset)
    printf( 'save done to %s\n', opts.dataset)
  else
    local map = {}
    map.train = totext(dataset.train, opts.train, '.aner')
    map.valid = totext(dataset.valid, opts.valid, '.aner')
    map.test = totext(dataset.test, opts.test, '.aner')
    
    local infile = opts.dataset
    local pos = infile:rfind('.')
    local outfile = infile:sub(1, pos-1) .. '.aner.map' .. infile:sub(pos)
    torch.save(outfile, map)
    print(outfile)
  end
  xprintln('time spend %fs', timer:time().real)
end

main()

