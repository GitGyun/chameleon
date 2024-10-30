exp_name=$1

cmd="python main.py --stage 1 --dataset fsc147 --exp_name ${exp_name} ${@:2:$#}"

echo $cmd
eval $cmd
