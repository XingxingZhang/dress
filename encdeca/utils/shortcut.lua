
function printf(s, ...)
  return io.write(s:format(...))
end

function xprint(s, ...)
  local ret = io.write(s:format(...))
  io.flush()
  return ret
end

function xprintln(s, ...)
  return xprint(s .. '\n', ...)
end

-- time --
function readableTime(stime)
  local intervals = {1, 60,  3600}
  local units = {"s", "min", "h"}
  local i = 2
  while i <= #intervals do
      if stime < intervals[i] then
        break
      end
      i = i + 1
  end
  return string.format( '%.2f%s', stime/intervals[i-1], units[i-1] )
end

-- for tables --
function table.keys(t)
  local ks = {}
  for k, _ in pairs(t) do
    ks[#ks + 1] = k
  end
  return ks
end

function table.len(t)
  local size = 0
  for _ in pairs(t) do
    size = size + 1
  end
  return size
end

-- for strings
function xtoCharSet(s)
  local set = {}
  local i = 1
  local ch = ''
  while true do
    ch = s:sub(i, i)
    if ch == '' then break end
    set[ch] = true
    i = i + 1
  end
  return set
end

function string.splitc(s, cseps)
  local strs = {}
  local cset = xtoCharSet(cseps)
  local i, ch = 1, ' '
  while ch ~= '' do
    while true do
      ch = s:sub(i, i)
      if ch == '' or not cset[ch] then break end 
      i = i + 1
    end
    local chs = {}
    while true do
      ch = s:sub(i, i)
      if ch == '' or cset[ch] then break end
      chs[#chs + 1] = ch
      i = i + 1
    end
    if #chs > 0 then strs[#strs + 1] = table.concat(chs) end
  end
  
  return strs
end

function string.starts(s, pat)
  return s:sub(1, string.len(pat)) == pat
end

function string.ends(s, pat)
  return pat == '' or s:sub(-string.len(pat)) == pat
end

function string.trim(s)
  return s:match'^()%s*$' and '' or s:match'^%s*(.*%S)'
end

function string.rfind(s, sub, istart, iend, isNotPlain)
  istart = 1 or istart
  iend = #s or iend
  if isNotPlain == nil then isNotPlain = false end
  local sub_ = sub:reverse()
  local pos1, pos2 = s:reverse():find(sub_, istart, iend, not isNotPlain)
  if pos1 ~= nil then
    return #s - pos2 + 1, #s - pos1 + 1
  end
end

-- the following is for arrays --
function table.extend(a, b)
  for _, v in ipairs(b) do
    a[#a + 1] = v
  end
  return a
end

function table.subtable(t, istart, iend)
  local N = #t
  assert(istart <= iend and istart >= 1 and iend <= N, 
    'invalid istart or iend')
  local subT = {}
  for i = istart, iend do
    subT[#subT + 1] = t[i]
  end
  
  return subT
end

function table.contains(t, key)
  for _, v in ipairs(t) do
    if v == key then return true end
  end
  return false
end

function table.find(t, key)
  for i, v in ipairs(t) do
    if v == key then return i end
  end
  return nil
end

function table.clear(t)
  for i, _ in ipairs(t) do
    t[i] = nil
  end
end

-- the following is for IOs --
function xreadlines(infile)
  local fin = io.open(infile, 'r')
  local lines = {}
  while true do
    local line = fin:read()
    if line == nil then break end
    lines[#lines + 1] = line
  end
  fin:close()
  
  return lines
end

function xcountlines(infile)
  local fin = io.open(infile, 'r')
  local cnt = 0
  while true do
    local line = fin:read()
    if line == nil then break end
    cnt = cnt + 1
  end
  fin:close()
  
  return cnt
end

function xmatches(s, reg)
  local istart, iend = s:find(reg)
  return istart == 1 and iend == s:len()
end





