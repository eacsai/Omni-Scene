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
    --work-dir /data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_all_z3

python train_mp3d_cylinder_double.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_double_pixel.py \
    --work-dir /data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_pixel_new

python train_mp3d_cylinder_double_volume.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_double_volume.py \
    --work-dir /data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_volume_z3

python train_mp3d_cylinder_double_volume.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_double_volume_high.py \
    --work-dir /data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_volume_high

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

python train_360Loc_cylinder_double.py \
    --py-config "configs/OmniScene/omni_gs_160x320_360Loc_cylinder_double_volume.py" \
    --work-dir "/data/qiwei/nips25/workdirs/omni_gs_160x320_360Loc_Cylinder_Double_Volume"

python train_360Loc_cylinder_double.py \
    --py-config "configs/OmniScene/omni_gs_160x320_360Loc_cylinder_double_pixel.py" \
    --work-dir "/data/qiwei/nips25/workdirs/omni_gs_160x320_360Loc_Cylinder_Double_Pixel"

python train_360Loc_cylinder_double_pixel.py \
    --py-config "configs/OmniScene/omni_gs_160x320_360Loc_cylinder_double_pixel.py" \
    --work-dir "/data/qiwei/nips25/workdirs/omni_gs_160x320_360Loc_Cylinder_Double_Pixel_new"

python train_360Loc_cylinder_double_pixel.py \
    --py-config "configs/OmniScene/omni_gs_160x320_360Loc_cylinder_double_pixel.py" \
    --work-dir "/data/qiwei/nips25/workdirs/omni_gs_160x320_360Loc_Cylinder_Double_Pixel_Pan"

python train_360Loc_cylinder_double_volume.py \
    --py-config "configs/OmniScene/omni_gs_160x320_360Loc_cylinder_double_volume.py" \
    --work-dir "/data/qiwei/nips25/workdirs/omni_gs_160x320_360Loc_Cylinder_Double_Volume_new"

python train_360Loc_cylinder_double_volume_pan.py \
    --py-config "configs/OmniScene/omni_gs_160x320_360Loc_cylinder_double_volume_pan.py" \
    --work-dir "/data/qiwei/nips25/workdirs/omni_gs_160x320_360Loc_Cylinder_Double_Volume_Pan"

python train_360Loc_cylinder_double_volume_pan2.py \
    --py-config "configs/OmniScene/omni_gs_160x320_360Loc_cylinder_double_volume_pan2.py" \
    --work-dir "/data/qiwei/nips25/workdirs/omni_gs_160x320_360Loc_Cylinder_Double_Volume_Pan2"

python train_360Loc_cylinder_double_volume_pan2.py \
    --py-config "configs/OmniScene/omni_gs_160x320_360Loc_cylinder_double_volume_pan2.py" \
    --work-dir "/data/qiwei/nips25/workdirs/omni_gs_160x320_360Loc_Cylinder_Double_Volume_Pan2_fix"

python train_360Loc_cylinder_double.py \
    --py-config "configs/OmniScene/omni_gs_160x320_360Loc_cylinder_double_all.py" \
    --work-dir "/data/qiwei/nips25/workdirs/omni_gs_160x320_360Loc_Cylinder_Double_All_Volume"

python train_360Loc.py \
    --py-config configs/OmniScene/omni_gs_160x320_360Loc.py \
    --work-dir /data/qiwei/nips25/workdirs/omni_gs_160x320_360Loc_Spherical


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
    --output-dir "/data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_all_z3" \
    --load-from "checkpoint-36000"

python evaluate_mp3d_double.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_double_pixel.py \
    --output-dir /data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_pixel_new \
    --load-from "checkpoint-36000"

python evaluate_mp3d_double.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_double_volume.py \
    --output-dir "/data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_volume_z3" \
    --load-from "checkpoint-36000"

python evaluate_mp3d_double.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_double_volume_high.py \
    --output-dir "/data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_volume_high" \
    --load-from "checkpoint-3000"

python evaluate_mp3d_double.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_double_all.py \
    --output-dir "/data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_all_z3" \
    --load-from "checkpoint-21000"

python evaluate_360Loc_double_all.py \
    --py-config "configs/OmniScene/omni_gs_160x320_360Loc_cylinder_double_pixel.py" \
    --output-dir "/data/qiwei/nips25/workdirs/omni_gs_160x320_360Loc_Cylinder_Double_Pixel_Pan" \
    --load-from "checkpoint-3000"

python evaluate_360Loc_double.py \
    --py-config "configs/OmniScene/omni_gs_160x320_360Loc_cylinder_double_volume.py" \
    --output-dir "/data/qiwei/nips25/workdirs/omni_gs_160x320_360Loc_Cylinder_Double_Volume_new" \
    --load-from "checkpoint-3000"

python evaluate_360Loc_double_all.py \
    --py-config "configs/OmniScene/omni_gs_160x320_360Loc_cylinder_double_volume_pan.py" \
    --output-dir "/data/qiwei/nips25/workdirs/omni_gs_160x320_360Loc_Cylinder_Double_Volume_Pan" \
    --load-from "checkpoint-3000"

python evaluate_360Loc_double_all.py \
    --py-config "configs/OmniScene/omni_gs_160x320_360Loc_cylinder_double_volume_pan2.py" \
    --output-dir "/data/qiwei/nips25/workdirs/omni_gs_160x320_360Loc_Cylinder_Double_Volume_Pan2" \
    --load-from "checkpoint-3000"

python evaluate_360Loc_double_Pan2.py \
    --py-config "configs/OmniScene/omni_gs_160x320_360Loc_cylinder_double_volume_pan2.py" \
    --output-dir "/data/qiwei/nips25/workdirs/omni_gs_160x320_360Loc_Cylinder_Double_Volume_Pan2_fix" \
    --load-from "checkpoint-3000"

python evaluate_360Loc_double.py \
    --py-config "configs/OmniScene/omni_gs_160x320_360Loc_cylinder_double_all.py" \
    --output-dir "/data/qiwei/nips25/workdirs/omni_gs_160x320_360Loc_Cylinder_Double_All_Volume" \
    --load-from "checkpoint-3000"