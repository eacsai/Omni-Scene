accelerate launch --config-file accelerate_config.yaml train_vigor.py \
    --py-config configs/OmniScene/omni_gs_nusc_novelview_r50_224x400.py \
    --work-dir workdirs/omni_gs_nusc_novelview_r50_224x400


python train_vigor.py \
    --py-config configs/OmniScene/omni_gs_cube_160x320.py \
    --work-dir workdirs/omni_gs_cube_160x320

# vigor panorama
python train_vigor.py \
    --py-config configs/OmniScene/omni_gs_160x320.py \
    --work-dir workdirs/omni_gs_160x320
# vigor cube
python train_vigor.py \
    --py-config configs/OmniScene/omni_gs_cube_160x320.py \
    --work-dir workdirs/omni_gs_cube_160x320

python train_mp3d.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d.py \
    --work-dir workdirs/omni_gs_160x320_mp3d_new

python train_mp3d.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_pixel.py \
    --work-dir workdirs/omni_gs_160x320_mp3d_pixel

python train_mp3d.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_pixel.py \
    --work-dir workdirs/omni_gs_160x320_mp3d_pixel_double

python train_mp3d.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_spherical.py \
    --work-dir /data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_Spherical

python train_mp3d.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_spherical_cross.py \
    --work-dir /data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_Spherical_cross

python train_mp3d_double.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_spherical_double.py \
    --work-dir /data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_Spherical_double

python train_mp3d_original_double.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_double.py \
    --work-dir /data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_double

python train_mp3d.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_spherical_decare.py \
    --work-dir /data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_Spherical_decare

python train_360Loc.py \
    --py-config configs/OmniScene/omni_gs_160x320_360Loc_pixel.py \
    --work-dir workdirs/omni_gs_160x320_360Loc_pixel


python evaluate_mp3d.py \
    --py-config "configs/OmniScene/omni_gs_160x320_mp3d_decare.py" \
    --output-dir "/data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_Spherical_decare" \
    --load-from "checkpoint-6000"

python evaluate_mp3d.py \
    --py-config "configs/OmniScene/omni_gs_160x320_mp3d.py" \
    --output-dir "/data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d" \
    --load-from "checkpoint-3000"

python evaluate_mp3d.py \
    --py-config "configs/OmniScene/omni_gs_160x320_mp3d_spherical_cross.py" \
    --output-dir "/data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_Spherical_cross" \
    --load-from "checkpoint-3000"

python evaluate_mp3d_double.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_spherical_double.py \
    --output-dir "/data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_Spherical_double" \
    --load-from "checkpoint-3000"

python evaluate_mp3d_double.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_double.py \
    --output-dir "/data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_double" \
    --load-from "checkpoint-3000"