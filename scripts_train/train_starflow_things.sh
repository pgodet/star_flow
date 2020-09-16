#!/bin/bash

# experiments and datasets meta
EXPERIMENTS_HOME="experiments"

# datasets
FLYINGTHINGS_HOME=(YOUR PATH)/FlyingThings3DSubset
SINTEL_HOME=(YOUR PATH)/mpisintelcomplete

# model and checkpoint
MODEL=StarFlow
EVAL_LOSS=MultiScaleEPE_PWC_Occ_video_upsample
CHECKPOINT=None
SIZE_OF_BATCH=4
NFRAMES=4
DEVICE=0

# save path
SAVE_PATH="$EXPERIMENTS_HOME/$MODEL-ftthings"

# training configuration
python ../main.py \
--batch_size=$SIZE_OF_BATCH \
--batch_size_val=$SIZE_OF_BATCH \
--checkpoint=$CHECKPOINT \
--lr_scheduler=MultiStepLR \
--lr_scheduler_gamma=0.5 \
--lr_scheduler_milestones="[257, 287, 307, 317]" \
--model=$MODEL \
--num_workers=6 \
--device=$DEVICE \
--optimizer=Adam \
--optimizer_lr=1e-4 \
--optimizer_weight_decay=4e-4 \
--save=$SAVE_PATH \
--start_epoch=217 \
--total_epochs=327 \
--training_augmentation=RandomAffineFlowOccVideo \
--training_augmentation_crop="[384,768]" \
--training_dataset=FlyingThings3dMultiframeCleanTrain \
--training_dataset_nframes=$NFRAMES \
--training_dataset_photometric_augmentations=True \
--training_dataset_root=$FLYINGTHINGS_HOME \
--training_key=total_loss \
--training_loss=$EVAL_LOSS \
--validation_dataset=FlyingThings3dMultiframeCleanTest \
--validation_dataset_nframes=$NFRAMES \
--validation_dataset_photometric_augmentations=False \
--validation_dataset_root=$FLYINGTHINGS_HOME \
--validation_key=epe \
--validation_loss=$EVAL_LOSS
