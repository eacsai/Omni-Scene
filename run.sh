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
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_spherical_cross_cos.py \
    --work-dir /data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_Spherical_cross_cos

python train_mp3d_cylinder.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder.py \
    --work-dir /data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_18000

python train_mp3d_cylinder_double_volume.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_double_all.py \
    --work-dir /data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_all

python train_mp3d_cylinder_double.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_double_pixel.py \
    --work-dir /data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_pixel_corssatten

python train_mp3d_cylinder_double_volume.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_double_volume.py \
    --work-dir /data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_volume

python train_mp3d_cylinder_volume.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_volume.py \
    --work-dir /data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_volume

python train_mp3d_cylinder_volume.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_pixel.py \
    --work-dir /data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_pixel

python train_mp3d.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_spherical_cross.py \
    --work-dir /data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_Spherical_cross

python train_mp3d_double.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_spherical_double_4.py \
    --work-dir /data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_Spherical_double_4

python train_mp3d_double.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_spherical_double_8inverse.py \
    --work-dir /data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_Spherical_double_8inverse

python train_mp3d_original_double.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_double.py \
    --work-dir /data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_double

python train_mp3d.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_spherical_decare.py \
    --work-dir /data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_Spherical_decare

python train_360Loc.py \
    --py-config configs/OmniScene/omni_gs_160x320_360Loc_pixel.py \
    --work-dir workdirs/omni_gs_160x320_360Loc_pixel


python train_360Loc.py \
    --py-config configs/OmniScene/omni_gs_160x320_360Loc.py \
    --work-dir /data/qiwei/nips25/workdirs/omni_gs_160x320_360Loc_Spherical

python train_360Loc.py \
    --py-config configs/OmniScene/omni_gs_160x320_360Loc_1.py \
    --work-dir /data/qiwei/nips25/workdirs/omni_gs_160x320_360Loc_Spherical_1

python evaluate_mp3d.py \
    --py-config "configs/OmniScene/omni_gs_160x320_mp3d_cylinder_pixel_volume.py" \
    --output-dir "/data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_pixel_volume" \
    --load-from "checkpoint-27000"

python evaluate_mp3d.py \
    --py-config "configs/OmniScene/omni_gs_160x320_mp3d_cylinder_volume_pixel.py" \
    --output-dir "/data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_volume_pixel" \
    --load-from "checkpoint-3000"

python evaluate_mp3d.py \
    --py-config "configs/OmniScene/omni_gs_160x320_mp3d.py" \
    --output-dir "/data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d" \
    --load-from "checkpoint-6000"

python evaluate_mp3d.py \
    --py-config "configs/OmniScene/omni_gs_160x320_mp3d_spherical_cross_conf.py" \
    --output-dir "/data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_Spherical_cross_conf" \
    --load-from "checkpoint-3000"

python evaluate_mp3d.py \
    --py-config "configs/OmniScene/omni_gs_160x320_mp3d_cylinder.py" \
    --output-dir "/data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_18000" \
    --load-from "checkpoint-3000"


python evaluate_mp3d.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_volume.py \
    --output-dir /data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_volume \
    --load-from "checkpoint-3000"

python evaluate_mp3d_double.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_double.py \
    --output-dir "/data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double" \
    --load-from "checkpoint-33000"

python evaluate_mp3d_volume.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_spherical_double_8.py \
    --output-dir "/data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_Spherical_double_8" \
    --load-from "checkpoint-3000"

python evaluate_mp3d_double.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_double.py \
    --output-dir "/data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_double" \
    --load-from "checkpoint-27000"

python evaluate_mp3d_double.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_double_all.py \
    --output-dir "/data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_all" \
    --load-from "checkpoint-15000"

python evaluate_mp3d_double.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_double_pixel.py \
    --output-dir /data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_pixel_corssatten \
    --load-from "checkpoint-15000"

python evaluate_mp3d_double.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_double_volume.py \
    --output-dir "/data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_volume" \
    --load-from "checkpoint-3000"

python evaluate_mp3d_double.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_double_all.py \
    --output-dir "/data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_all" \
    --load-from "checkpoint-3000"

python evaluate_360Loc.py \
    --py-config "configs/OmniScene/omni_gs_160x320_360Loc_1.py" \
    --output-dir "/data/qiwei/nips25/workdirs/omni_gs_160x320_360Loc_Spherical_1" \
    --load-from "checkpoint-18000"