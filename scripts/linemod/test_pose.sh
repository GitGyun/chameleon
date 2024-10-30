exp_name=$1
class_name=$2
exp_subname="task:pose_6d_class:${class_name}_shot:50_is:224_lr:0.005_sid:0"
exp_subname_coord="task:segment_semantic_class:${class_name}_shot:50_is:224_lr:0.005_sid:0"
coord_path="${exp_name}/${exp_subname_coord}/logs/bbox_${class_name}.npy"

cmd="python main.py --stage 2 --dataset linemod --exp_name ${exp_name} --class_name ${class_name} --exp_subname ${exp_subname} --coord_path ${coord_path} ${@:3:$#}"

echo $cmd
eval $cmd
