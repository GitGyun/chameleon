exp_name=$1
exp_subname="task:cellpose_shot:50_is:224_lr:0.003_sid:0"

cmd="python main.py --stage 2 --dataset cellpose --exp_name ${exp_name} --exp_subname ${exp_subname} ${@:2:$#}"

echo $cmd
eval $cmd
