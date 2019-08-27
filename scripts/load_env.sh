source ~/python-venv/iccv/bin/activate
export PYTHONPATH=$PYTHONPATH:/cluster/home/buehlmar/iccv_projects/SPADE_custom/
module load hdf5/1.10.1 python_gpu/3.6.4
cd /cluster/home/buehlmar/iccv_projects/SPADE_custom


export DATE=$(date +"%y%m%d")

export no_ganFeat_loss=

for use_vae in "" --use_vae
do
    export name=""$DATE"_spade""${use_vae//--/__}"${no_ganFeat_loss//--/__}""
    bsub -n 1 -W 24:00 -o "lsf_$name" -R "rusage[mem=16048, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" python train.py \
        --name $name \
         --dataset_mode openeds \
         --dataroot /cluster/work/hilliges/buehlmar/iccv/data/openeds/all.h5 \
         --aspect_ratio 0.8 \
         --no_instance \
         --load_size 256 \
         --crop_size 256 \
         --batchSize 1 \
         --preprocess_mode fixed \
         --no_vgg_loss \
         --tf_log \
         --no_html \
         $use_vae \
         --continue_train
done
export DATE=$(date +"%y%m%d")

export no_ganFeat_loss=

for use_vae in "" --use_vae
do
    export name=""$DATE"_spade""${use_vae//--/__}"${no_ganFeat_loss//--/__}""
    bsub -n 1 -W 24:00 -o "lsf_$name" -R "rusage[mem=16048, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" python train.py \
        --name $name \
         --dataset_mode openeds \
         --dataroot /cluster/work/hilliges/buehlmar/iccv/data/openeds/all.h5 \
         --aspect_ratio 0.8 \
         --no_instance \
         --load_size 256 \
         --crop_size 256 \
         --batchSize 1 \
         --preprocess_mode fixed \
         --no_vgg_loss \
         --tf_log \
         --no_html \
         $use_vae \
         --continue_train
done




