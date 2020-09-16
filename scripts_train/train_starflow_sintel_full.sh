#!/bin/bash

# experiments and datasets meta
EXPERIMENTS_HOME="experiments"

# datasets
SINTEL_HOME=(YOUR PATH)/mpisintelcomplete

# model and checkpoint
MODEL=StarFlow
EVAL_LOSS=MultiScaleEPE_PWC_Occ_video_upsample_Sintel
CHECKPOINT=None
SIZE_OF_BATCH=4
NFRAMES=4
DEVICE=0

# save path
SAVE_PATH="$EXPERIMENTS_HOME/$MODEL-ftsintel1-full"

# training configuration
python ../main.py \
--batch_size=$SIZE_OF_BATCH \
--batch_size_val=$SIZE_OF_BATCH \
--checkpoint=$CHECKPOINT \
--lr_scheduler=MultiStepLR \
--lr_scheduler_gamma=0.5 \
--lr_scheduler_milestones="[89, 130, 170, 190, 195, 200, 220, 240, 260, 280]" \
--model=$MODEL \
--num_workers=6 \
--device=$DEVICE \
--optimizer=Adam \
--optimizer_lr=1.5e-05 \
--optimizer_weight_decay=4e-4 \
--save=$SAVE_PATH \
--start_epoch=1 \
--total_epochs=300 \
--training_augmentation=RandomAffineFlowOccVideo \
--training_augmentation_crop="[384,768]" \
--training_dataset=SintelMultiframeTrainingCombFull \
--training_dataset_nframes=$NFRAMES \
--training_dataset_photometric_augmentations=True \
--training_dataset_root=$SINTEL_HOME \
--training_key=total_loss \
--training_loss=$EVAL_LOSS \
--validation_dataset=SintelMultiframeTrainingFinalValid \
--validation_dataset_nframes=$NFRAMES \
--validation_dataset_photometric_augmentations=False \
--validation_dataset_root=$SINTEL_HOME \
--validation_key=epe \
--validation_loss=$EVAL_LOSS

# save path
SAVE_PATH_2="$EXPERIMENTS_HOME/$MODEL-ftsintel2-full"

# training configuration
python ../main.py \
--batch_size=$SIZE_OF_BATCH \
--batch_size_val=$SIZE_OF_BATCH \
--checkpoint=$SAVE_PATH \
--lr_scheduler=MultiStepLR \
--lr_scheduler_gamma=0.5 \
--lr_scheduler_milestones="[481, 562, 643, 683, 693, 703, 743, 783, 824, 864]" \
--model=$MODEL \
--num_workers=6 \
--device=$DEVICE \
--optimizer=Adam \
--optimizer_lr=1e-05 \
--optimizer_weight_decay=4e-4 \
--save=$SAVE_PATH_2 \
--start_epoch=301 \
--total_epochs=451 \
--training_augmentation=RandomAffineFlowOccVideo \
--training_augmentation_crop="[384,768]" \
--training_dataset=SintelMultiframeTrainingFinalFull \
--training_dataset_nframes=$NFRAMES \
--training_dataset_photometric_augmentations=True \
--training_dataset_root=$SINTEL_HOME \
--training_key=total_loss \
--training_loss=$EVAL_LOSS \
--validation_dataset=SintelMultiframeTrainingFinalValid \
--validation_dataset_nframes=$NFRAMES \
--validation_dataset_photometric_augmentations=False \
--validation_dataset_root=$SINTEL_HOME \
--validation_key=epe \
--validation_loss=$EVAL_LOSS
