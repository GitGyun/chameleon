exp_name=$1
class_names=( "ape" "benchviseblue" "cam" "can" "cat" "driller" "duck" "eggbox" "glue" "holepuncher" "iron" "lamp" "phone" )
for class_name in ${class_names[@]};
do
    bash scripts/linemod/finetune.sh ${exp_name} ${class_name} ${@:2:$#}
    bash scripts/linemod/test.sh ${exp_name} ${class_name} ${@:2:$#}
done