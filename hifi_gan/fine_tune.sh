### Shell script to run fine-tuning of HiFi-GAN. The only argument is the suffix of the experiment directory.

# First, copy pretrained/UNIVERSAL_V1 to cp_hifigan_$1 based on the experiment name 
cp -r pretrained/UNIVERSAL_V1 cp_hifigan_$1

# Then, run training 
python train.py \
    --input_wavs_dir ft_dataset_$1/wavs \
    --input_mels_dir ft_dataset_$1/mels \
    --input_training_file ft_dataset_$1/train_filelist.txt \
    --input_validation_file ft_dataset_$1/val_filelist.txt \
    --checkpoint_path cp_hifigan_$1 \
    --config config_v1.json \
    --fine_tuning True 