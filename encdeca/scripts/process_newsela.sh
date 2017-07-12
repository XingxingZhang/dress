
train=/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/newsela/all/all.train
valid=/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/newsela/all/all.valid
test=/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/newsela/all/all.test
dataset=/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/newsela/V0V4_V1V4_V2V4_V3V4.t7
dataset=/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/newsela/all/all.t7
th anonymize_ner.lua --train $train --valid $valid --test $test --dataset $dataset --totext


# train=/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/newsela/V0V4_V1V4_V2V4_V3V4.aner.train
train=/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/newsela/all/all.aner.train
# valid=/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/newsela/V0V4_V1V4_V2V4_V3V4.aner.valid
valid=/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/newsela/all/all.aner.valid
# test=/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/newsela/V0V4_V1V4_V2V4_V3V4.aner.test
test=/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/newsela/all/all.aner.test
# map=/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/newsela/V0V4_V1V4_V2V4_V3V4.aner.map.t7
map=/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/newsela/all/all.aner.map.t7
th recover_anonymous.lua --train $train --valid $valid --test $test --map $map

