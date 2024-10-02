exp_name=$1
support_idx=$2
exp_subname="task:segment_medical_shot:20_is:384_lr:0.001_sid:${support_idx}"

cmd="python main.py --stage 2 --dataset isic2018 --exp_name ${exp_name} --support_idx ${support_idx} --exp_subname ${exp_subname} ${@:3:$#}"

echo $cmd
eval $cmd
