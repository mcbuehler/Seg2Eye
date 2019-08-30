source ~/python-venv/iccv/bin/activate
export PYTHONPATH=$PYTHONPATH:/cluster/home/buehlmar/iccv_projects/SPADE_custom/
module load hdf5/1.10.1 python_gpu/3.6.4
cd /cluster/home/buehlmar/iccv_projects/SPADE_custom


export DATE=$(date +"%y%m%d")

export use_vae=
export no_ganFeat_loss=
export load_size=256
export l2=15
export l1=0
export spadeStyleGen=--spadeStyleGen
export z_dim=256
export w_dim=8
export bs=4
export lr=0.0002


for l1 in 0
do
    for l2 in 15 20
    do
        for load_size in 256
        do
            for bs in 2 4
            do
                export lr_adapt=$( echo "($bs * $lr) " | bc -l )
                export name=""$DATE"_size"$load_size"_bs"$bs"_lr"$lr_adapt"_l1_"$l1"_l2_"$l2"_z"$z_dim"_w"$w_dim"_${use_vae//--/__}"${no_ganFeat_loss//--/__}""${spadeStyleGen//--/__}""
                echo $name
                bsub -n 1 -W 24:00 -o "lsf_$name" -R "rusage[mem=16048, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" python train.py \
                    --dataroot /cluster/work/hilliges/buehlmar/iccv/data/openeds/all.h5 \
                    --name $name \
                     --load_size $load_size \
                     --crop_size $load_size \
                     --batchSize $bs \
                     --tf_log \
                     --lambda_l1 $l1 \
                     --lambda_l2 $l2 \
                     --z_dim $z_dim \
                     --w_dim $w_dim \
                     --lr $lr_adapt \
                     $spadeStyleGen \
                     $use_vae
            done
        done
    done
done




export name=
# TEST
for DK in train validation test
do
    bsub -n 1 -W 4:00 -R "rusage[mem=16048, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" \
        python test.py --name $name --dataset_mode openeds \
        --dataroot /cluster/work/hilliges/buehlmar/iccv/data/openeds/all.h5  --dataset_key $DK \
         --aspect_ratio 0.8 --no_instance --load_size 256 --crop_size 256 \
         --preprocess_mode fixed --batchSize 24 --write_error_log \
         --netG spade --use_vae
done




