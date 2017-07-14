
train=/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/newsela/all/V0V4_V1V4_V2V4_V3V4_V0V3_V0V2_V1V3.train
valid=/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/newsela/all/V0V4_V1V4_V2V4_V3V4_V0V3_V0V2_V1V3.valid
test=/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/newsela/all/V0V4_V1V4_V2V4_V3V4_V0V3_V0V2_V1V3.test
dataset=/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/newsela/all/V0V4_V1V4_V2V4_V3V4_V0V3_V0V2_V1V3.t7
th anonymize_ner.lua --train $train --valid $valid --test $test --dataset $dataset --totext


# train=/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/newsela/V0V4_V1V4_V2V4_V3V4.aner.train
train=/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/newsela/all/V0V4_V1V4_V2V4_V3V4_V0V3_V0V2_V1V3.aner.train
# valid=/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/newsela/V0V4_V1V4_V2V4_V3V4.aner.valid
valid=/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/newsela/all/V0V4_V1V4_V2V4_V3V4_V0V3_V0V2_V1V3.aner.valid
# test=/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/newsela/V0V4_V1V4_V2V4_V3V4.aner.test
test=/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/newsela/all/V0V4_V1V4_V2V4_V3V4_V0V3_V0V2_V1V3.aner.test
# map=/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/newsela/V0V4_V1V4_V2V4_V3V4.aner.map.t7
map=/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/newsela/all/V0V4_V1V4_V2V4_V3V4_V0V3_V0V2_V1V3.aner.map.t7
th recover_anonymous.lua --train $train --valid $valid --test $test --map $map

