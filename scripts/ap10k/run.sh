exp_name=$1
class_name=$2

bash scripts/ap10k/finetune.sh $exp_name $class_name ${@:3:$#}
bash scripts/ap10k/test.sh $exp_name $class_name ${@:3:$#}
