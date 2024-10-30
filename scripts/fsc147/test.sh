exp_name=$1
exp_subname="task:object_counting_shot:50_is:512_lr:0.001_sid:0"

cmd="python main.py --stage 2 --dataset fsc147 --exp_name ${exp_name} --exp_subname ${exp_subname} ${@:2:$#}"

echo $cmd
eval $cmd
