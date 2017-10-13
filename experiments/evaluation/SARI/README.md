# How to get *corpus-level* SARI scores?
1. Download Joshua SImplification System described in Xu et al, 2016 [here](https://drive.google.com/file/d/0B1P1xW5xNISsdXdoX1RQNmVSSkE/view?usp=sharing)
2. copy ``star`` and ``star_1`` to ``ppdb-simplification-release-joshua5.0/joshua/bin``
```
unzip ppdb-simplification-release-joshua5.0.zip
cp star star_1 ppdb-simplification-release-joshua5.0/joshua/bin
```
3. set JAVA_HOME and JOSHUA environments.
```
export JAVA_HOME=/usr/lib/jvm/java-1.8.0
export JOSHUA=/disk/scratch/Software/joshua_5/ppdb-simplification-release-joshua5.0/joshua

export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
```
4. evaluate SARI
```
cd ppdb-simplification-release-joshua5.0/joshua/bin
./star output reference src      # for 8 references evaluate (e.g., wikilarge)
./star_1 output reference src    # for single reference evaluate (e.g., newsela and wikismall)
```
## A note for WikiLarge
You should turn your system output into lower case (you can use ``../BLEU/tolower.py``), since the wikilarge test set in Xu et al., (2016) is lower-cased.

Then you can use the sample script (``wikilarge.show_all_sari.sh``) below to get sari scores
```
sysdir=dress/all-system-output/WikiLarge/test/lower
input=/disk/scratch/Dataset/simplification/data/turkcorpus/test.8turkers.tok.norm
ref=/disk/scratch/Dataset/simplification/data/turkcorpus/test.8turkers.tok.turk

for sysout in `ls $sysdir`
do
	./star $sysdir/$sysout $ref $input
	echo "====================="
	echo $sysout
	echo "\n\n\n"
done

```
After run the following command,
```
sh wikilarge.show_all_sari.sh | grep ======== -B 2 -A 2
```
you should get the following output
```
STAR = 0.3708210095744638

=====================
Dress.lower

--
STAR = 0.37266058818588216

=====================
Dress-Ls.lower

--
STAR = 0.3565754396121206

=====================
EncDecA.lower

--
STAR = 0.3139665078989411

=====================
Hybrid.lower

--
STAR = 0.38558843050332037

=====================
PBMT-R.lower

--
STAR = 0.39964857928109127

=====================
SBMT-SARI.lower


```

# References
Wei Xu, Courtney Napoles, Ellie Pavlick, Quanze Chen, and Chris Callison-Burch. 2016. Optimizing statistical machine translation for text simplification.
Transactions of the Association for Computational Linguistics, 4:401–415.
