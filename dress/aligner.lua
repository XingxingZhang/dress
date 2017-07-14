
require '.'
require 'EncDecAWE'
require 'BiText_Dataset'

local Aligner = torch.class('NeuralAligner')

function Aligner:__init(modelPath)
  local statePath = modelPath:sub(1, -3) .. 'state.t7'
  local opts = torch.load(statePath)
  self.opts = opts
  self.encdec = EncDecAWE(opts)
  self.encdec:load(modelPath)
  -- extract attention model
  self.attentions = {}
  for t = 1, self.opts.seqLen do
    local gmap = BModel.get_module_map({self.encdec.enc_lstms[t], self.encdec.dec_lstms[t]})
    self.attentions[t] = gmap.attention
  end
end

-- input: trainFile (or validFile or testFile)
function Aligner:generateAlignment(trainFile)
  local dataIter = BiText_Dataset.createBatch(self.opts.src_vocab, self.opts.dst_vocab,
    trainFile, self.opts.validBatchSize)
  local cnt = 0
  local timer = torch.Timer()
  local sents = {}
  for x, x_mask, y in dataIter do
    local y_mask = y:ne(0)
    local xlens = x_mask:sum(1):view(-1)
    self.encdec:validBatch(x, x_mask, y)
    local Tx, Ty, bs = x:size(1), y:size(1), x:size(2)
    local sents_info = {}
    local bs_cnt = 0
    for i = 1, bs do
      local xlen = xlens[i]
      local ylen = y_mask[{ {}, i }]:sum() - 1
      if ylen > 0 then
        sents_info[i] = {align = torch.LongTensor(ylen):fill(0), 
          detailed_align = torch.FloatTensor(ylen, xlen):fill(0)}
        bs_cnt = bs_cnt + 1
      else
        break
      end
    end
    
    for t = 1, Ty - 1 do
      local att = self.attentions[t].output:float()
      for i = 1, bs_cnt do
        if y_mask[{t + 1, i}] ~= 0 then
          sents_info[i].detailed_align[{t, {}}] = att[{ i, {1, xlens[i]} }]
          local _, idx = att[{ i, {1, xlens[i]} }]:max(1)
          sents_info[i].align[{t}] = idx[1]
        end
      end
    end
    
    for i = 1, bs do
      sents[#sents + 1] = sents_info[i]
    end
    
    cnt = cnt + 1
    if cnt % 10 == 0 then
      collectgarbage()
      xprint('\r[%s] cnt = %d (%s)', basename(trainFile), cnt, readableTime(timer:time().real))
    end
  end
  
  xprintln('\r[%s] cnt = %d (%s)', basename(trainFile), cnt, readableTime(timer:time().real))
  xprintln('#sents = %d', #sents)
  
  local sent_cnt = 0
  local fin_src = io.open(trainFile .. '.src')
  local fin_dst = io.open(trainFile .. '.dst')
  while true do
    sent_cnt = sent_cnt + 1
    local sline = fin_src:read()
    local dline = fin_dst:read()
    if sline == nil then
      assert(dline == nil, 'MUST have the same number of lines')
      break
    end
    local src_words = sline:splitc(' \t')
    local dst_words = dline:splitc(' \t')
    local sent = sents[sent_cnt]
    sent.src = src_words
    sent.dst = dst_words
    if #dst_words + 1 ~= sent.align:size(1) then
      print(dst_words)
      print(sent.align)
    end
    if #src_words + 1 ~= sent.detailed_align:size(2) then
      print(src_words)
      print(sent.detailed_align)
    end
  end
  
  return sents
end

function Aligner.showAlign(alignPath)
  xprintln('visualize alignment')
  local dataset = torch.load(alignPath)
  xprintln('load alignment dataset done!')
  local output = alignPath .. '.vis'
  local fout = io.open(output, 'w')
  
  local function showSingle(train_split, label)
    fout:write(string.format('==%s==\n', label))
    
    for cnt, sent in ipairs(train_split) do
      fout:write(string.format('id = %d\n', cnt))
      fout:write(string.format('source = %s\n', table.concat(sent.src, ' ')))
      fout:write(string.format('target = %s\n', table.concat(sent.dst, ' ')))
      local src_words = {'###eos###'}
      for i = #sent.src, 1, -1 do
        src_words[#src_words + 1] = sent.src[i]
      end
      local dst_words = {'###eos###'}
      for i = 1, #sent.dst do
        dst_words[#dst_words + 1] = sent.dst[i]
      end
      dst_words[#dst_words + 1] = '###eos###'
      local dalign = sent.detailed_align
      for i = 1, #dst_words - 1 do
        fout:write(string.format('< %s\n', dst_words[i]))
        local att = dalign[{i, {}}]
        local as, idxs = att:sort(1, true)
        for j = 1, as:size(1) do
          local src_i = idxs[j]
          local src_a = as[j]
          fout:write(string.format('%d %s %f\n', src_i, src_words[src_i], src_a))
        end
        fout:write(string.format('> %s\n\n', dst_words[i+1]))
      end
      
      fout:write('\n\n')
      
      if cnt % 1000 == 0 or cnt == #train_split then
        xprintln('[%s] %d', label, cnt)
      end
    end
    
  end
  
  showSingle(dataset.train, 'train')
  showSingle(dataset.valid, 'valid')
  showSingle(dataset.test, 'test')
  
  fout:close()
  xprintln('visualize done!')
end

local function main()
  local cmd = torch.CmdLine()
  cmd:option('--modelPath', '/disk/scratch1/xingxing.zhang/seq2seq/sent_simple/encdec_lm_tm/encdec_basline/model_0.001.256.2L.we.full.2l.ft0.t7',
    'model path')
  cmd:option('--train', '/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/newsela/all/V0V4_V1V4_V2V4_V3V4_V0V3_V0V2_V1V3.aner.train', 
    'train file')
  cmd:option('--valid', 
    '/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/newsela/all/V0V4_V1V4_V2V4_V3V4_V0V3_V0V2_V1V3.aner.valid', 
    'valid file')
  cmd:option('--test', 
    '/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/newsela/all/V0V4_V1V4_V2V4_V3V4_V0V3_V0V2_V1V3.aner.test', 
    'test file')
  cmd:option('--output', '/disk/scratch1/xingxing.zhang/seq2seq/sent_simple/encdec_lm_neu_tm/align/newsela.align', 'save path')
  cmd:option('--showAlign', false, 'show alignment information')
  local opts = cmd:parse(arg)
  
  if not paths.filep(opts.output) then
    require 'cutorch'
    require 'cunn'
    
    local aligner = NeuralAligner(opts.modelPath)
    local align_data = {}
    align_data.train = aligner:generateAlignment(opts.train)
    align_data.valid = aligner:generateAlignment(opts.valid)
    align_data.test = aligner:generateAlignment(opts.test)
    xprintln('save %s ...', opts.output)
    torch.save(opts.output, align_data)
    xprintln('save %s!', opts.output)
  else
    xprintln('%s exists!', opts.output)
  end
  
  if opts.showAlign then
    NeuralAligner.showAlign(opts.output)
  end
  
end

if not package.loaded['aligner'] then
  main()
end

