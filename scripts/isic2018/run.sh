exp_name=$1
support_idx=$2

bash scripts/isic2018/finetune.sh $exp_name $support_idx ${@:3:$#}
bash scripts/isic2018/test.sh $exp_name $support_idx ${@:3:$#}
