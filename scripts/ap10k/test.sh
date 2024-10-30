exp_name=$1
class_name=$2
exp_subname="task:animalkp_class:${class_name}_shot:20_is:224_lr:0.001_sid:0"

cmd="python main.py --stage 2 --dataset ap10k --exp_name ${exp_name} --class_name ${class_name} --exp_subname ${exp_subname} ${@:3:$#}"

echo $cmd
eval $cmd
