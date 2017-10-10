# BLEU evaluation on WikiLarge dataset
I will use the EncDecA model as an example.

1. Download Joshua Simplification System described in Xu et al, 2016 from [here](https://drive.google.com/file/d/0B1P1xW5xNISsdXdoX1RQNmVSSkE/view?usp=sharing)
```
unzip ppdb-simplification-release-joshua5.0.zip
```
2. Download the 8 references test set
```
git clone https://github.com/cocoxu/simplification
```
3. Get system output and change it into lower-case (note references are also lower-cased)
```
cp ../../../all-system-output/WikiLarge/test/EncDecA .
python tolower.py EncDecA EncDecA.lower
```
Note that you can also run this script to get the sys output of EncDecA ``https://github.com/XingxingZhang/dress/blob/better-eval/experiments/wikilarge/encdeca/generate/run_std.sh``
<br>

4. Evaluate BLEU
```
./ppdb-simplification-release-joshua5.0/joshua/bin/bleu EncDecA.lower ./simplification/data/turkcorpus/test.8turkers.tok.turk 8
```
You will get a bleu score of 0.8885.
```
BLEU_precision(1) = 7399 / 7719 = 0.9585
BLEU_precision(2) = 6768 / 7360 = 0.9196
BLEU_precision(3) = 6219 / 7004 = 0.8879
BLEU_precision(4) = 5688 / 6649 = 0.8555
BLEU_precision = 0.9046

Length of candidate corpus = 7719
Effective length of reference corpus = 7857
BLEU_BP = 0.9823

  => BLEU = 0.8885
```

# References
Wei Xu, Courtney Napoles, Ellie Pavlick, Quanze Chen, and Chris Callison-Burch. 2016. Optimizing statistical machine translation for text simplification.
Transactions of the Association for Computational Linguistics, 4:401–415.
