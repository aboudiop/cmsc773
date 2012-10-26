#!/bin/sh
PROJ_PATH="$HOME/Dev/773proj"
export PATH="$PROJ_PATH/maxent:$PROJ_PATH/feateng:$PROJ_PATH/eval:$PATH"
export PYTHONPATH="$PROJ_PATH:$PYTHONPATH"

# prepare input
tmp="megam.run"
cp train.megam "$tmp"
echo "DEV" >> "$tmp"
cat dev.megam >> "$tmp"

# training
maxent_train 1 1.5 2 2.5 3 3.5 4 4.5 5 5.5 6 6.5 7 7.5 8 8.5 9 9.5 10 < "$tmp" > "$tmp.weights"
# testing
maxent_predict "$tmp.weights" < test.megam | megam_kbest.py > "$tmp.out"

# evaluation
num_to_code.py map.megam < test.megam > test.labels
num_to_code.py map.megam < megam.run.out > megam.run.out.labels
accuracy.py test.labels megam.run.out.labels > megam.run.eval
conf_mat.py test.labels megam.run.out.labels > megam.run.csv

# k-best evaluation
for k in `seq 10`; do
    accuracy.py test.megam megam.run.out $k | grep Overall
done | grep -oE '[0-9]+\.[0-9]+' | cat -n > megam.run.kbest