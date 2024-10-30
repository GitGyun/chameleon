exp_name=$1
class_name=$2
lr=$3

cmd="python main.py --stage 1 --dataset davis2017 --exp_name ${exp_name} --class_name ${class_name} --lr ${lr} ${@:4:$#}"

echo $cmd
eval $cmd
