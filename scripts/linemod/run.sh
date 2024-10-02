exp_name=$1
class_name=$2

bash scripts/linemod/run_segment.sh $exp_name $class_name ${@:3:$#}
bash scripts/linemod/run_pose.sh $exp_name $class_name ${@:3:$#}
