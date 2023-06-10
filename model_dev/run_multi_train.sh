# take a target vinput variable

# loop though 0 to 270, with 1 step size
for target in $(seq 0 270)
do
    python3 train.py --target $target
done