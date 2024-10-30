exp_name=$1
class_name=$2

bash scripts/davis2017/finetune.sh $exp_name $class_name ${@:3:$#}
bash scripts/davis2017/test.sh $exp_name $class_name ${@:3:$#}
