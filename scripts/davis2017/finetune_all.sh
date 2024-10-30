exp_name=$1
class_names=( "bike-packing" "blackswan" "bmx-trees" "breakdance" "camel" "car-roundabout" "car-shadow" "cows" "dance-twirl" "dog" "dogs-jump" "drift-chicane" "drift-straight" "goat" "gold-fish" "horsejump-high" "india" "judo" "kite-surf" "lab-coat" "libby" "loading" "mbike-trick" "motocross-jump" "paragliding-launch" "parkour" "pigs" "scooter-black" "shooting" "soapbox" )
for class_name in ${class_names[@]};
do
    # bash scripts/davis2017/finetune.sh ${exp_name} ${class_name} ${@:2:$#}
    echo scripts/davis2017/finetune.sh ${exp_name} ${class_name} ${@:2:$#}
done
