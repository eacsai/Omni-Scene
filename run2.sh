### random
# train
python train_mp3d_cylinder_double_random.py \
     --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_pixel.py \
     --work-dir /home/qiwei/ICLR25/workdirs/omni_gs_160x320_mp3d_cylinder_double_pixel_random
python train_mp3d_cylinder_double_random.py \
     --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_volume.py \
     --work-dir /data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_volume_random
python train_mp3d_cylinder_double_random.py \
     --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_all.py \
     --work-dir /data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_all_random


# eval
python evaluate_mp3d_double.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_volume.py \
    --output-dir /data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_volume_random \
    --load-from "checkpoint-36000"

python evaluate_mp3d.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_volume.py \
    --output-dir /data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_volume_random \
    --load-from "checkpoint-36000"

python evaluate_mp3d.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_pixel.py \
    --output-dir /home/qiwei/ICLR25/workdirs/omni_gs_160x320_mp3d_cylinder_double_pixel_random \
    --load-from "checkpoint-45000"

python evaluate_mp3d.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_all.py \
    --output-dir /data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_all_random \
    --load-from "checkpoint-45000"

python evaluate_mp3d_double.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_all.py \
    --output-dir /data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_all_random \
    --load-from "checkpoint-54000"

### double
# train
python train_mp3d_cylinder_double.py \
     --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_pixel.py \
     --work-dir /home/qiwei/ICLR25/workdirs/omni_gs_160x320_mp3d_cylinder_double_pixel

python train_mp3d_cylinder_double.py \
     --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_volume.py \
     --work-dir /data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_volume

python train_mp3d_cylinder_double.py \
     --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_all.py \
     --work-dir /data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_all

# eval

python evaluate_mp3d_double.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_pixel.py \
    --output-dir /home/qiwei/ICLR25/workdirs/omni_gs_160x320_mp3d_cylinder_double_pixel \
    --load-from "checkpoint-36000"


python evaluate_mp3d_double.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_volume.py \
    --output-dir /data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_volume \
    --load-from "checkpoint-36000"