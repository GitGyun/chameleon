exp_name=$1

bash scripts/fsc147/finetune.sh $exp_name ${@:2:$#}
bash scripts/fsc147/test.sh $exp_name ${@:2:$#}
