exp_name=$1
support_idx=$2

cmd="python main.py --stage 1 --dataset isic2018 --exp_name ${exp_name} --support_idx ${support_idx} ${@:3:$#}"

echo $cmd
eval $cmd
