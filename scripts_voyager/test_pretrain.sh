torchrun --nproc_per_node 8 \
   -m training.main   --save-frequency 1     --save-most-recent     --zeroshot-frequency 1    \
   --train-data '/voyager/datasets/coco/mscoco/{00000..00059}.tar'     --dataset-type webdataset     \
   --lr "2.048e-3"     --beta1 0.9     --beta2 0.95     --warmup 782     --wd 0.2     --batch-size 256     --epochs=1     --workers=0  \
   --model RN50       --force-image-size 84        --log-every-n-steps 32     --seed 0     --logs ./logs/     --imagenet-val '/voyager/datasets/imagenet/ImageNet/val'   \
   --report-to "wandb"  --wandb-project-name "clip_2024" --hanaba --name "name21" --train-num-samples 2000 --dataset-resampled