#!/bin/bash

# experiments and datasets meta
EXPERIMENTS_HOME="experiments"

# datasets
KITTI_HOME=(YOUR PATH)/KittiComb

# model and checkpoint
MODEL=StarFlow
EVAL_LOSS=MultiScaleEPE_PWC_Occ_upsample_KITTI
CHECKPOINT=None
SIZE_OF_BATCH=4
NFRAMES=4
DEVICE=0

# save path
SAVE_PATH="$EXPERIMENTS_HOME/$MODEL-ftkitti-full"

# training configuration
python ../main.py \
--batch_size=$SIZE_OF_BATCH \
--batch_size_val=$SIZE_OF_BATCH \
--checkpoint=$CHECKPOINT \
--lr_scheduler=MultiStepLR \
--lr_scheduler_gamma=0.5 \
--lr_scheduler_milestones="[456, 659, 862, 963, 989, 1014, 1116, 1217, 1319, 1420]" \
--model=$MODEL \
--num_workers=6 \
--device=$DEVICE \
--optimizer=Adam \
--optimizer_lr=3e-05 \
--optimizer_weight_decay=4e-4 \
--save=$SAVE_PATH \
--start_epoch=1 \
--total_epochs=550 \
--training_augmentation=RandomAffineFlowOccVideoKitti \
--training_augmentation_crop="[320,896]" \
--training_dataset=KittiMultiframeCombFull \
--training_dataset_nframes=$NFRAMES \
--training_dataset_photometric_augmentations=True \
--training_dataset_root=$KITTI_HOME \
--training_dataset_preprocessing_crop=True \
--training_key=total_loss \
--training_loss=$EVAL_LOSS \
--validation_dataset=KittiMultiframeComb2015Val \
--validation_dataset_nframes=$NFRAMES \
--validation_dataset_photometric_augmentations=True \
--validation_dataset_root=$KITTI_HOME \
--validation_dataset_preprocessing_crop=True \
--validation_key=epe \
--validation_loss=$EVAL_LOSS
