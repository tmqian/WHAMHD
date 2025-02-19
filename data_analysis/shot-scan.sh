# Reads all shots from a given date, Executes script in batches of 10
# TQ 2/18/2025

day=251218000
script=9-multi

# tens digit (put 5 if you have 59 shots)
for i in $(seq 0 5)
do
    # ones digit 
    for j in $(seq 0 9)
    do
        # calculate shot number
        shot=$(($day + 10*$i + $j))

        # Insert your data plot script here
        cmd="python3 ${script}.py ${shot}"
        echo $cmd
        $cmd  &
    done

    # Let everyone finish before starting the next batch
    wait

done
