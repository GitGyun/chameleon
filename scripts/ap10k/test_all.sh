exp_name=$1
class_names=( "Antelope" "Cat" "Elephant" "Giraffe" "Hippo" "Horse" "Mouse" "Pig" )
for class_name in ${class_names[@]};
do
    bash scripts/ap10k/test.sh ${exp_name} ${class_name} ${@:2:$#}
done
