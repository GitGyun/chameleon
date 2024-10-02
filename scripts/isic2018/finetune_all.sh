exp_name=$1
support_idxs=( "0" "1" "2" "3" "4" )
for support_idx in ${support_idxs[@]};
do
    bash scripts/ap10k/finetune.sh ${exp_name} ${support_idx} ${@:2:$#}
done
