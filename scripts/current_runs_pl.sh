


--name test_pl3 --dataroot /home/marcel/projects/data/openeds/all.h5 --tf_log --batchSize 64 --niter 1 --lr 0.00001 --weight_decay 0.01
export DATE=190908
export bs=64


# runs with l2 10 and 15 and bs1 and cmseq and w8 and 32
for lr in 0.000001 0.00001 0.00004
do
    for wc in 0 0.001 0.01
    do
        for radam in "" --use_radam
        do
            export name=""$DATE"_pupil_bs"$bs"_lr"$lr"_wc"$wc"${radam//--/__}"
            echo $name
            bsub -n 2 -W 24:00 -o "lsf_"$name"" -R "rusage[mem=16048, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" python train_pupillocator.py \
                            --dataroot /cluster/work/hilliges/buehlmar/iccv/data/openeds/all.h5 \
                            --name $name \
                             --batchSize $bs \
                             --tf_log \
                             --lr $lr \
                             --weight_decay $wc --continue_train \
                             $radam
         done
    done
done


