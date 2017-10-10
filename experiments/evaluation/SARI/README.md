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
3. evaluate SARI
```
cd ppdb-simplification-release-joshua5.0/joshua/bin
./star output reference src      # for 8 references evaluate (e.g., wikilarge)
./star_1 output reference src    # for single reference evaluate (e.g., newsela and wikismall)
```

# References
Wei Xu, Courtney Napoles, Ellie Pavlick, Quanze Chen, and Chris Callison-Burch. 2016. Optimizing statistical machine translation for text simplification.
Transactions of the Association for Computational Linguistics, 4:401–415.
