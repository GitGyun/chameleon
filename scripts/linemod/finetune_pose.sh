exp_name=$1
class_name=$2

cmd="python main.py --stage 1 --dataset linemod --exp_name ${exp_name} --class_name ${class_name} ${@:3:$#}"

echo $cmd
eval $cmd
