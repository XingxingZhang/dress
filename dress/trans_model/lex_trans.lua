
include '../utils/shortcut.lua'

local LexTrans = torch.class('LexTransModel')

function LexTrans:__init(selfTransDiscount)
  self.selfTransDiscount = selfTransDiscount or 1.0
end

function LexTrans:load(src_vocab_path, dst_vocab_path, lex_file)
  self.src_vocab = type(src_vocab_path) == 'string' and torch.load( src_vocab_path ) or src_vocab_path
  self.dst_vocab = type(dst_vocab_path) == 'string' and torch.load( dst_vocab_path ) or dst_vocab_path
  xprintln('src nvocab = %d, dst nvocab = %d', self.src_vocab.nvocab, self.dst_vocab.nvocab)
  
  local function get_wid(vocab, wd)
    local wid = vocab.word2idx[wd]
    if wid == nil then
      return vocab.UNK, true
    else
      return wid, false
    end
  end
  
  self.lex_trans_tbl = torch.FloatTensor(self.src_vocab.nvocab, self.dst_vocab.nvocab):fill(0)
  local fin = io.open(lex_file)
  local cnt = 0
  local src_unk_cnt = 0
  local dst_unk_cnt = 0
  local src_null = {}
  while true do
    local line = fin:read()
    if line == nil then break end
    cnt = cnt + 1
    local fields = line:trim():splitc(' \t')
    assert(#fields == 3, 'must have 3 fields!!!')
    local dst_wd, src_wd, prob = fields[1], fields[2], tonumber(fields[3])
    local dst_wid, dst_is_unk = get_wid(self.dst_vocab, dst_wd)
    local src_wid, src_is_unk = get_wid(self.src_vocab, src_wd)
    if src_is_unk then
      src_unk_cnt = src_unk_cnt + 1
    else
      if dst_wd == src_wd then
        if self.selfTransDiscount ~= 1 then
          prob = prob * self.selfTransDiscount
        end
      end
      
      self.lex_trans_tbl[{src_wid, dst_wid}] = self.lex_trans_tbl[{src_wid, dst_wid}] + prob
      if dst_is_unk then
        if dst_wd ~= 'NULL' then
          dst_unk_cnt = dst_unk_cnt + 1
          -- xprintln('dst is unk: %s', line)
        else
          -- note when dst_wd is NULL, then don't count the prob
          self.lex_trans_tbl[{src_wid, dst_wid}] = self.lex_trans_tbl[{src_wid, dst_wid}] - prob
          if src_null[src_wid] == nil then
            src_null[src_wid] = prob
          else
            error('impossible')
          end
        end
      end
    end
  end
  
  xprintln('cnt = %d, src cnt = %d, rate = %f', cnt, src_unk_cnt, src_unk_cnt/cnt)
  xprintln('cnt = %d, dst cnt = %d, rate = %f', cnt, dst_unk_cnt, dst_unk_cnt/cnt)
  
  local okayRate = 0
  local zeroRate = 0
  local nullRate = 0
  local nullProbSum, nullProbMax, nullProbMin = 0, 0, 123
  local max_diff = 1e-3
  for i = 1, self.lex_trans_tbl:size(1) do
    if math.abs( self.lex_trans_tbl[{i, {}}]:sum() - 1 ) < max_diff then
      okayRate = okayRate + 1
    end
    if self.lex_trans_tbl[{i, {}}]:sum() == 0 then zeroRate = zeroRate + 1 end
    if src_null[i] then 
      nullRate = nullRate + 1
      nullProbSum = nullProbSum + src_null[i]
      nullProbMax = math.max(nullProbMax, src_null[i])
      nullProbMin = math.min(nullProbMin, src_null[i])
    end
  end
  xprintln('okay rate = %f', okayRate/self.lex_trans_tbl:size(1))
  xprintln('zero rate = %f', zeroRate/self.lex_trans_tbl:size(1))
  xprintln('null rate = %f', nullRate/self.lex_trans_tbl:size(1))
  xprintln('null prob average = %f, max = %f, min = %f', nullProbSum/nullRate, nullProbMax, nullProbMin)
end

function LexTrans:lookup_word_index(wids, show_msg)
  
  if wids:dim() == 1 then
    if show_msg then
      for i = 1, wids:size(1) do
        local wid = wids[i]
        print( self.src_vocab.idx2word[wid] )
      end
    end
    
    return self.lex_trans_tbl:index(1, wids)
  elseif wids:dim() == 2 then
    if show_msg then
      for i = 1, wids:size(1) do
        for j = 1, wids:size(2) do
          local wid = wids[{i, j}]
          print( self.src_vocab.idx2word[wid] )
        end
        print ''
      end
    end
    
    local output = self.lex_trans_tbl:index(1, wids:view(-1))
    
    return output:view(wids:size(1), wids:size(2), output:size(2))
  else
    error('only support 1D and 2D tensor')
  end
end

function LexTrans:lookup_word(wds, show_msg)
  local function wd2wid(wds, vocab)
    local wids = {}
    for i = 1, #wds do
      if type(wds[i]) == 'table' then
        wids[i] = wd2wid(wds[i], vocab)
      else
        local wid = vocab.word2idx[wds[i]] or vocab.UNK
        wids[i] = wid
      end
    end
    return wids
  end
  
  local wids = torch.LongTensor( wd2wid(wds, self.src_vocab) )
  if show_msg then
    print(wids)
  end
  
  return self:lookup_word_index(wids, show_msg)
  
end

local function main()
  local src_vocab = '/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/newsela/all/V0V4_V1V4_V2V4_V3V4_V0V3_V0V2_V1V3.aner.train.src.vocab.tmp.t7'
  local dst_vocab = '/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/newsela/all/V0V4_V1V4_V2V4_V3V4_V0V3_V0V2_V1V3.aner.train.dst.vocab.tmp.t7'
  local lex_file = '/disk/scratch/Software/SMT_moses/sent_simple_ner/work/train/model/lex.f2e'
  
  local lex_trans = LexTransModel()
  lex_trans:load(src_vocab, dst_vocab, lex_file)
  
  local wids = torch.LongTensor({1, 3, 5})
  local output = lex_trans:lookup_word_index(wids, true)
  print(output:sum(2))
  
  wids = torch.LongTensor({{1, 3, 5}, {2, 4, 6}})
  output = lex_trans:lookup_word_index(wids, true)
  print(output:size())
  print(output:sum(3))
  
  -- local wds = {{'a', 'A'}, {'the', 'The'}}
  local wds = {'a', 'A', 'the', 'The'}
  print(wds)
  lex_trans:lookup_word(wds, true)
end

if not package.loaded['lex_trans'] then
	main()
end

