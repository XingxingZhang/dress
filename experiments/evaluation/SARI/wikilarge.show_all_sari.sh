
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


