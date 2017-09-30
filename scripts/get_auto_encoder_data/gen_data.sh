
ori_data_dir=../wiki.full.aner

cat $ori_data_dir.train.src $ori_data_dir.train.dst > train.tmp
cat $ori_data_dir.valid.src $ori_data_dir.valid.dst > valid.tmp
cat $ori_data_dir.test.src $ori_data_dir.test.dst > test.tmp

uniq.py train.tmp train.u.tmp
uniq.py valid.tmp valid.txt
uniq.py test.tmp test.txt


python filter_train.py --train train.u.tmp --valid valid.txt --out train.1.tmp
python filter_train.py --train train.1.tmp --valid test.txt --out train.2.tmp

shuffle.py train.2.tmp -r 123 -o train.txt

rm *.tmp


python check.py

