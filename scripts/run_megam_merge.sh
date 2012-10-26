#!/bin/sh
PROJ_PATH="$HOME/Dev/773proj"
export PATH="$PROJ_PATH/tools:$PROJ_PATH/coarsefine:$PROJ_PATH/feateng:$PROJ_PATH/eval:$PATH"
export PYTHONPATH="$PROJ_PATH:$PYTHONPATH"

# grouping
for i in train dev test; do
    merge_labels.py megam.groups $i.group map.megam < $i.megam > $i.group.all
done

for i in "all" `seq 0 8`; do
    # prepare input
    tmp="megam.run.$i"
    cp train.group.$i "$tmp"
    echo "DEV" >> "$tmp"
    cat dev.group.$i >> "$tmp"
    echo "TEST" >> "$tmp"
    cat test.group.$i >> "$tmp"

    # training
    megam -maxi 500 -lambda 100 -tune multiclass "$tmp" > "megam.weights.$i"
done

# clean up
rm megam.run.* {train,test,dev}.group*

# tagging
for i in "all" `seq 0 8`; do
    megam -predict "megam.weights.$i" multiclass test.megam 2> /dev/null > "megam.out.$i"
done

# final output
predict_merged.py map.megam megam.groups megam.out.all megam.out.[0-8] | megam_kbest.py > megam.run.out

# k-best evaluation
for k in `seq 10`; do
    accuracy.py test.megam megam.run.out $k | grep Overall
done | grep -oE '[0-9]+\.[0-9]+' | cat -n > megam.run.kbest