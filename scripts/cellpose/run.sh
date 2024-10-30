exp_name=$1

bash scripts/cellpose/finetune.sh $exp_name ${@:2:$#}
bash scripts/cellpose/test.sh $exp_name ${@:2:$#}
