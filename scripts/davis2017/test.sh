exp_name=$1
class_name=$2
lr=$3
snptf=$4
exp_subname="task:vos_class:${class_name}_shot:1_is:384_lr:${lr}_sid:0${snptf}"

cmd="python main.py --stage 2 --dataset davis2017 --exp_name ${exp_name} --class_name ${class_name} --exp_subname ${exp_subname} --result_dir results_davis2017_lr:${lr}${snptf} ${@:5:$#}"

echo $cmd
eval $cmd
