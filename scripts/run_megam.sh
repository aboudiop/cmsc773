#!/bin/sh

PROJ_PATH=~/Dev/773proj
export PATH="$PROJ_PATH/tools:$PROJ_PATH/feateng:$PROJ_PATH/eval:$PATH"
export PYTHONPATH="$PROJ_PATH:$PYTHONPATH"

# prepare input
tmp="megam.run"
cp train.megam "$tmp"
echo "DEV" >> "$tmp"
cat dev.megam >> "$tmp"
echo "TEST" >> "$tmp"
cat test.megam >> "$tmp"

# training
megam -maxi 500 -lambda 100 -tune multiclass "$tmp" > "$tmp.weights"
# testing
megam -predict "$tmp.weights" multiclass test.megam | megam_kbest.py > "$tmp.out"

# evaluation
num_to_code.py map.megam < test.megam > test.labels
num_to_code.py map.megam < megam.run.out > megam.run.out.labels
accuracy.py test.labels megam.run.out.labels > megam.run.eval
conf_mat.py test.labels megam.run.out.labels > megam.run.csv

# k-best evaluation
for k in `seq 10`; do
    ../../eval/accuracy.py test.megam megam.run.out $k | grep Overall
done | grep -oE '[0-9]+\.[0-9]+' | cat -n > megam.run.kbest