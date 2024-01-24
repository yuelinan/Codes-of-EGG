python train_free_event.py \
  --gpu_id=1 \
  --model_name=Event_Free_Generation \
  --lr=1e-4 \
  --max_len=512 \
  --types=train \
  --epochs=5 \
  --batch_size=8 \
  --class_num=62 \

# python train_event.py \
#   --gpu_id=1 \
#   --model_name=Event_Generation_cat \
#   --lr=1e-4 \
#   --max_len=512 \
#   --types=train \
#   --epochs=5 \
#   --batch_size=8 \
#   --class_num=62 \

