source ~/python-venv/iccv/bin/activate
export PYTHONPATH=$PYTHONPATH:/cluster/home/buehlmar/iccv_projects/SPADE_custom/
module load hdf5/1.10.1 python_gpu/3.6.4
cd /cluster/home/buehlmar/iccv_projects/SPADE_custom
module load eth_proxy


export DATE=$(date +"%y%m%d")

export load_size=256
export w_dim=1024
export lr=0.0002
export checkpoints_dir=./checkpoints
export cm=add

export radam=
export DK=train
export DR=/cluster/work/hilliges/buehlmar/iccv/data/openeds/190910_all.h5
export wc=0
export bs=1
export spadeStyleGen=--spadeStyleGen
export lr=0.0002
export norm_G=spectralspadesyncbatch3x3
export SSM=ref_random6
export SAM=max
export netG=spaderefiner
export l2=15
export l1=0
export lambda_w=0.5
export lambda_feat=0.001
export NS=3
export SEG=/cluster/work/hilliges/buehlmar/iccv/data/openeds/190914_deeplab_seg_predictions.h5
export SR=datasets/distances_and_indices.h5
export PT=--pretrainD
export pretrained_path=/cluster/home/buehlmar/iccv_projects/SPADE_custom/checkpoints/190912_size512_bs1_lr.0002_l1_0_l2_15_lambda_w_0.5_lambda_feat_0.001_w1024___spadeStyleGen_cmadd_ns3_SAMmax_wc0_SSMref_random6
export openeds=0
export l1=0
export l2=0

for PT in "" --pretrainD
do
    for lr in 0.00004 0.0002
    do
        for l2 in 10 15
        do
            export lr_adapt=$( echo "($bs * $lr) " | bc -l )
            export name=""$DATE"_refiner_size"$load_size"_bs"$bs"_lr"$lr_adapt"_l1_"$l1"_l2_"$l2"_lopeneds_"$openeds"_lambda_w_"$lambda_w"_lambda_feat_"$lambda_feat"_w"$w_dim"_"${spadeStyleGen//--/__}""${PT//--/__}"_cm"$cm"_ns"$NS"_SAM"$SAM"_wc"$wc"_SSM"$SSM""

            echo $name

            bsub -n 4 -W 24:00 -J $name -o "lsf_"$DK"_"$name"" -R "rusage[mem=16048, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]"  \
            python train.py \
                                                --dataroot $DR \
                                                --name $name \
                                                --dataset_key $DK \
                                                --netG $netG \
                                                 --load_size $load_size \
                                                 --crop_size $load_size \
                                                 --batchSize $bs \
                                                 --tf_log \
                                                 --input_nc 2 \
                                                 --lambda_l1 $l1 \
                                                 --lambda_l2 $l2 \
                                                 --lambda_openeds $openeds \
                                                 --lambda_style_w $lambda_w \
                                                 --lambda_style_feat  $lambda_feat \
                                                 --w_dim $w_dim \
                                                 --lr $lr_adapt \
                                                 $spadeStyleGen \
                                                 --style_sample_method $SSM \
                                                 --combine_mode $cm \
                                                 --input_ns $NS \
                                                 --style_aggr_method $SAM \
                                                 --weight_decay $wc\
                                                 --norm_G $norm_G \
                                                 --style_ref $SR \
                                                 --seg_file $SEG \
                                                 $PT --pretrained_path $pretrained_path
        done
    done
done



export l2=0
export l1=0
export openeds=0

for PT in "" --pretrainD
do
    for lr in 0.00004 0.0002
    do
        for l1 in 10 15
        do
            export lr_adapt=$( echo "($bs * $lr) " | bc -l )
            export name=""$DATE"_refiner_size"$load_size"_bs"$bs"_lr"$lr_adapt"_l1_"$l1"_l2_"$l2"_lopeneds_"$openeds"_lambda_w_"$lambda_w"_lambda_feat_"$lambda_feat"_w"$w_dim"_"${spadeStyleGen//--/__}""${PT//--/__}"_cm"$cm"_ns"$NS"_SAM"$SAM"_wc"$wc"_SSM"$SSM""

            echo $name

            bsub -n 4 -W 24:00 -J $name -o "lsf_"$DK"_"$name"" -R "rusage[mem=16048, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]"  \
            python train.py \
                                                --dataroot $DR \
                                                --name $name \
                                                --dataset_key $DK \
                                                --netG $netG \
                                                 --load_size $load_size \
                                                 --crop_size $load_size \
                                                 --batchSize $bs \
                                                 --tf_log \
                                                 --input_nc 2 \
                                                 --lambda_l1 $l1 \
                                                 --lambda_l2 $l2 \
                                                 --lambda_openeds $openeds \
                                                 --lambda_style_w $lambda_w \
                                                 --lambda_style_feat  $lambda_feat \
                                                 --w_dim $w_dim \
                                                 --lr $lr_adapt \
                                                 $spadeStyleGen \
                                                 --style_sample_method $SSM \
                                                 --combine_mode $cm \
                                                 --input_ns $NS \
                                                 --style_aggr_method $SAM \
                                                 --weight_decay $wc\
                                                 --norm_G $norm_G \
                                                 --style_ref $SR \
                                                 --seg_file $SEG \
                                                 $PT --pretrained_path $pretrained_path
        done
    done
done







export l2=0
export l1=0


for PT in "" --pretrainD
do
    for lr in 0.00004 0.0002
    do
        for openeds in 10 15
        do
            export lr_adapt=$( echo "($bs * $lr) " | bc -l )
            export name=""$DATE"_refiner_size"$load_size"_bs"$bs"_lr"$lr_adapt"_l1_"$l1"_l2_"$l2"_lopeneds_"$openeds"_lambda_w_"$lambda_w"_lambda_feat_"$lambda_feat"_w"$w_dim"_"${spadeStyleGen//--/__}""${PT//--/__}"_cm"$cm"_ns"$NS"_SAM"$SAM"_wc"$wc"_SSM"$SSM""

            echo $name

            bsub -n 4 -W 24:00 -J $name -o "lsf_"$DK"_"$name"" -R "rusage[mem=16048, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]"  \
            python train.py \
                                                --dataroot $DR \
                                                --name $name \
                                                --dataset_key $DK \
                                                --netG $netG \
                                                 --load_size $load_size \
                                                 --crop_size $load_size \
                                                 --batchSize $bs \
                                                 --tf_log \
                                                 --input_nc 2 \
                                                 --lambda_l1 $l1 \
                                                 --lambda_l2 $l2 \
                                                 --lambda_openeds $openeds \
                                                 --lambda_style_w $lambda_w \
                                                 --lambda_style_feat  $lambda_feat \
                                                 --w_dim $w_dim \
                                                 --lr $lr_adapt \
                                                 $spadeStyleGen \
                                                 --style_sample_method $SSM \
                                                 --combine_mode $cm \
                                                 --input_ns $NS \
                                                 --style_aggr_method $SAM \
                                                 --weight_decay $wc\
                                                 --norm_G $norm_G \
                                                 --style_ref $SR \
                                                 --seg_file $SEG \
                                                 $PT --pretrained_path $pretrained_path
        done
    done
done
















### TEST
bsub -n 4 -W 4:00 -J validate_dilated  -R "rusage[mem=16048, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]"  \
 python test.py --name 190915_refiner_size256_bs1_lr.0002_l1_10_l2_0_lopeneds_0_lambda_w_0.5_lambda_feat_0.001_w1024___spadeStyleGen_cmadd_ns3_SAMmax_wc0_SSMref_random6 \
  --dataroot $DR --style_ref $SR --seg_file $SEG --load_from_opt_file --dataset_key validation --write_error_log





# TRAIN with random 200
export DK=train
export SSM=ref_random200
export l2=15
export DATE=190916



for SSM in ref_random200 ref_random100 ref_random50
do
    export name=""$DATE"_refiner_size"$load_size"_bs"$bs"_lr"$lr_adapt"_l1_"$l1"_l2_"$l2"_lopeneds_"$openeds"_lambda_w_"$lambda_w"_lambda_feat_"$lambda_feat"_w"$w_dim"_"${spadeStyleGen//--/__}""${PT//--/__}"_cm"$cm"_ns"$NS"_SAM"$SAM"_wc"$wc"_SSM"$SSM""
    echo $name
    bsub -n 4 -W 120:00 -J $name -o "lsf_"$DK"_"$name"" -R "rusage[mem=16048, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]"  \
    python train.py \
                                        --dataroot $DR \
                                        --name $name \
                                        --dataset_key $DK \
                                        --netG $netG \
                                         --load_size $load_size \
                                         --crop_size $load_size \
                                         --batchSize $bs \
                                         --tf_log \
                                         --input_nc 2 \
                                         --lambda_l1 $l1 \
                                         --lambda_l2 $l2 \
                                         --lambda_openeds $openeds \
                                         --lambda_style_w $lambda_w \
                                         --lambda_style_feat  $lambda_feat \
                                         --w_dim $w_dim \
                                         --lr $lr \
                                         $spadeStyleGen \
                                         --style_sample_method $SSM \
                                         --combine_mode $cm \
                                         --input_ns $NS \
                                         --style_aggr_method $SAM \
                                         --weight_decay $wc\
                                         --norm_G $norm_G \
                                         --style_ref $SR \
                                         --seg_file $SEG \
                                         $PT --pretrained_path $pretrained_path \
                                         --dilate_test \
                                         --continue_train
done


export DK=train
for name in 190916_refiner_size256_bs1_lr_l1_0_l2_15_lopeneds_0_lambda_w_0.5_lambda_feat_0.001_w1024___spadeStyleGen__pretrainD_cmadd_ns3_SAMmax_wc0_SSMref_random100 \
    190916_refiner_size256_bs1_lr_l1_0_l2_15_lopeneds_0_lambda_w_0.5_lambda_feat_0.001_w1024___spadeStyleGen__pretrainD_cmadd_ns3_SAMmax_wc0_SSMref_random200 \
    190916_refiner_size256_bs1_lr_l1_0_l2_15_lopeneds_0_lambda_w_0.5_lambda_feat_0.001_w1024___spadeStyleGen__pretrainD_cmadd_ns3_SAMmax_wc0_SSMref_random50
do
    echo $name
    bsub -n 4 -W 4:00 -R "rusage[mem=16048, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]"  \
    python test.py \
    --name $name --dataroot $DR --style_ref $SR --seg_file $SEG \
     --load_from_opt_file --dataset_key $DK --style_sample_method ref_first --produce_npy
    bsub -n 4 -W 4:00 -J $name -o "lsf_"$DK"_"$name"" -R "rusage[mem=16048, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]"  \
    python test.py \
    --name $name --dataroot $DR --style_ref $SR --seg_file $SEG \
     --load_from_opt_file --dataset_key $DK --style_sample_method ref_first
done