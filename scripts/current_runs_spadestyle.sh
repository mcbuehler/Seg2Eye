# Cluster specific
source ~/python-venv/iccv/bin/activate
export PYTHONPATH=$PYTHONPATH:/cluster/home/buehlmar/iccv_projects/SPADE_custom/
module load hdf5/1.10.1 python_gpu/3.6.4
cd /cluster/home/buehlmar/iccv_projects/SPADE_custom
module load eth_proxy

export DR=/cluster/work/hilliges/buehlmar/iccv/data/openeds/190910_all.h5
export SEG=/cluster/work/hilliges/buehlmar/iccv/data/openeds/190914_deeplab_seg_predictions.h5
export SR=datasets/distances_and_indices.h5
export pretrained_path=/cluster/home/buehlmar/iccv_projects/SPADE_custom/checkpoints/190912_size512_bs1_lr.0002_l1_0_l2_15_lambda_w_0.5_lambda_feat_0.001_w1024___spadeStyleGen_cmadd_ns3_SAMmax_wc0_SSMref_random6


# Desktop specific
source ~/python-venv/pytorch_gpu/bin/activate
cd /home/marcel/projects/Seg2Eye
export PYTHONPATH=$PYTHONPATH:$(pwd)
export DR=/home/marcel/projects/data/openeds/190910_all.h5
export SR=/home/marcel/projects/data/openeds/datasets/distances_and_indices.h5




export DATE=$(date +"%y%m%d")
export load_size=256
export lr=0.0002
export checkpoints_dir=./checkpoints
export cm=add

export DK=train
export wc=0
export bs=1
export spadeStyleGen=--spadeStyleGen
export lr=0.0002
export norm_G=spectralspadesyncbatch3x3
export SSM=ref_random6
export SAM=max
export netG=spadestyle
export lambda_w=0.5
export lambda_feat=0.001
export NS=40
export openeds=0
export l1=0
export l2=15
export w_dim=16


export DATE=190924
# Runs with high dimensionality of w
export w_dim=1024
for SSM in ref_random6 ref_random200 ref_random100 ref_random50
do
    export name=""$DATE"_"$netG"_size"$load_size"_l2_"$l2"_lambda_w_"$lambda_w"_lambda_feat_"$lambda_feat"_w"$w_dim"_"${spadeStyleGen//--/__}"_cm"$cm"_ns"$NS"_SAM"$SAM"_SSM"$SSM""
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
                                         --input_nc 1 \
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
                                         --style_ref $SR
done



# Runs with low dimensionality of w
export w_dim=16
for SSM in ref_random6 ref_random200 ref_random100 ref_random50
do
    export name=""$DATE"_"$netG"_size"$load_size"_l2_"$l2"_lambda_w_"$lambda_w"_lambda_feat_"$lambda_feat"_w"$w_dim"_"${spadeStyleGen//--/__}"_cm"$cm"_ns"$NS"_SAM"$SAM"_SSM"$SSM""
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
                                         --input_nc 1 \
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
                                         --style_ref $SR
done


# Runs with low dimensionality of w, but large loss on feature maps
export w_dim=16
export lambda_feat=1
for SSM in ref_random6 ref_random200 ref_random100 ref_random50
do
    export name=""$DATE"_"$netG"_size"$load_size"_l2_"$l2"_lambda_w_"$lambda_w"_lambda_feat_"$lambda_feat"_w"$w_dim"_"${spadeStyleGen//--/__}"_cm"$cm"_ns"$NS"_SAM"$SAM"_SSM"$SSM""
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
                                         --input_nc 1 \
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
                                         --style_ref $SR
done



# Runs with low dimensionality of w, but large loss on feature gram maps
export DATE=190925
export w_dim=16
export SSM=ref_random100
for lg in 1000 10000 100000
do
    export name=""$DATE"_"$netG"_size"$load_size"_l2_"$l2"_lambda_w_"$lambda_w"_lambda_feat_"$lambda_feat"_lambda_gram_"$lg"_w"$w_dim"_"${spadeStyleGen//--/__}"_cm"$cm"_ns"$NS"_SAM"$SAM"_SSM"$SSM""
    echo $name
    bsub -n 4 -W 4:00 -J $name -o "lsf_"$DK"_"$name"" -R "rusage[mem=16048, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]"  \
    python train.py \
                                        --dataroot $DR \
                                        --name $name \
                                        --dataset_key $DK \
                                        --netG $netG \
                                         --load_size $load_size \
                                         --crop_size $load_size \
                                         --batchSize $bs \
                                         --tf_log \
                                         --input_nc 1 \
                                         --lambda_l1 $l1 \
                                         --lambda_l2 $l2 \
                                         --lambda_openeds $openeds \
                                         --lambda_style_w $lambda_w \
                                         --lambda_style_feat  $lambda_feat \
                                         --lambda_gram  $lg \
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
                                         --continue_train
done




export DK=test
export name=190925_spadestyle_size256_l2_15_lambda_w_0.5_lambda_feat_0.001_lambda_gram_100000_w16___spadeStyleGen_cmadd_ns4_SAMmax_SSMref_random100
bsub -n 1 -W 4:00 -o "lsf_"$DK"_""$name" -R "rusage[mem=32048, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" \
python test.py --dataroot $DR --name $name --dataset_key $DK --batchSize 1 --load_from_opt_file --produce_npy


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