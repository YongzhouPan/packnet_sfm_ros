for SEQ in 00 01 02 03 04 05 06 07 08 09 10
do
    python3 scripts/infer.py --checkpoint trained_models/PackNet01_MR_velsup_CStoK.ckpt --input /data/datasets/kitti_odom_color/sequences/${SEQ}/image_2/ --output depth_infer/${SEQ}_80 --save png
done