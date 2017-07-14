
include '../utils/shortcut.lua'
-- require 'shortcut'

local py = require('fb.python')
local pySari = py.import('sari.SARI')

local Sari = torch.class('SARI')

function Sari.SARIsent(src_sent, trg_sent, ref_sents)
	return py.eval( pySari.SARIsent(src_sent, trg_sent, ref_sents) )
end


function Sari.SARIsent_single(src_sent, trg_sent, ref_sent)
	return py.eval( pySari.SARIsent(src_sent, trg_sent, {ref_sent}) )
end


function Sari.SARIfile(src_file, trg_file, ref_file)
	local function file2lines(infile)
		local fin = io.open(infile)
		local lines = {}

		while true do
			local line = fin:read()
			if line == nil then break end
			lines[#lines + 1] = line
		end

		return lines
	end

	local score, cnt = 0, 0
	local src_sents = file2lines(src_file)
	local trg_sents = file2lines(trg_file)
	local ref_sents = file2lines(ref_file)

	xprintln('#src = %d, #trg = %d, #ref = %d', #src_sents, #trg_sents, #ref_sents)

	for i = 1, #src_sents do
		local src = src_sents[i]
		local trg = trg_sents[i]
		local ref = ref_sents[i]
		score = score + Sari.SARIsent_single(src, trg, ref)
		cnt = cnt + 1
	end

	return score / cnt
end

function Sari.getDynBatch(x, x_mask, y, y_pred, reward, reward_mask, 
    src_vocab, dst_vocab, sari_rev_weight)
  
  if sari_rev_weight == nil then
    sari_rev_weight = 0.5
  end
  
  reward:zero()
  reward_mask:zero()
  
  local ori_sents = { src = {}, dst = {}, ref = {} }
  
  local function get_word(vocab, wid)
    return vocab.idx2word[wid]
  end
  local batchSize = x:size(2)
  -- get source
  local src_sents = {}
  local dst_sents = {}
  local dst_pred_sents = {}
  for i = 1, batchSize do
    -- get source
    local src = {}
    local src_len = x_mask[{ {}, i }]:sum()
    for j = src_len, 2, -1 do
      table.insert(src, get_word(src_vocab, x[{ j, i }]))
    end
    -- get target
    local dst = {}
    for j = 2, y:size(1) do
      if y[{ j, i }] == dst_vocab.EOS or y[{ j, i }] == 0 then
        break
      else
        table.insert(dst, get_word(dst_vocab, y[{ j, i }]))
      end
    end
    -- get predict
    local dst_pred = {}
    local last_pos = -1
    local ended = false
    for j = 2, y_pred:size(1) do
      if y_pred[{ j, i }] == dst_vocab.EOS then
        last_pos = j
        ended = true
        reward_mask[{ j, i }] = 1
        break
      else
        table.insert(dst_pred, get_word(dst_vocab, y_pred[{ j, i }]))
        last_pos = j
        reward_mask[{ j, i }] = 1
      end
    end
    local src_sent = table.concat(src, ' ')
    local trg_sent = table.concat(dst_pred, ' ')
    local ref_sent = table.concat(dst, ' ')
    
    local r = 0
    if #dst ~= 0 then
      --[[
      -- r = Sari.SARIsent_single(src_sent, trg_sent, ref_sent)
      r = Sari.SARIsent_single(src_sent, ref_sent, trg_sent)
      --]]
      local sari = Sari.SARIsent_single(src_sent, trg_sent, ref_sent)
      local sari_reverse = Sari.SARIsent_single(src_sent, ref_sent, trg_sent)
      -- r = (sari + sari_reverse) / 2
      r = (1 - sari_rev_weight) * sari + sari_rev_weight * sari_reverse
      
      if r == 0 then r = 1e-5 end
      assert(last_pos ~= -1, 'last_pos must be valid!')
      reward[{ last_pos, i }] = r
      
      if #dst_pred == 0 then
        xprintln('src = \"%s\"', src_sent)
        xprintln('dst = \"%s\"', trg_sent)
        xprintln('ref = \"%s\"', ref_sent)
        xprintln('last pos = %d', last_pos)
      end
      
      table.insert(ori_sents.src, src_sent)
      table.insert(ori_sents.dst, trg_sent)
      table.insert(ori_sents.ref, ref_sent)
    end
  end
  
  return reward, reward_mask, ori_sents
end


local function main()
	local cmd = torch.CmdLine()
	cmd:option('--src', '/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/newsela/all/V0V4_V1V4_V2V4_V3V4_V0V3_V0V2_V1V3.aner.ori.valid.src', 'src file')
	cmd:option('--trg', '', 'trg file')
	cmd:option('--ref', '/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/newsela/all/V0V4_V1V4_V2V4_V3V4_V0V3_V0V2_V1V3.aner.ori.valid.dst', 'ref file')
	local opts = cmd:parse(arg)

	local sari = SARI()
	xprintln( 'SARI = %f', sari.SARIfile(opts.src, opts.trg, opts.ref) )
end

if not package.loaded['SARI'] then
	main()
else
	print '[SARI] loaded as package!'
end


