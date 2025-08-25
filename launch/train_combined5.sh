export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

mpirun -np 2 python train_combined5.py \
    --train_data_list data/train_list.txt --val_data_list data/val_list.txt \
    --feat_dim 144 --num_joints 52 \
    --num_samples 2560 --vertex_samples 2048 \
    --batch_size 24 --epochs 300 \
    --optimizer adam --learning_rate 2e-4 --weight_decay 1e-4 \
    --skeleton_loss_weight 1.0 --skin_loss_weight 0.5 \
    --skel_alpha 0.6 --skel_beta 0.8 --bone_len_weight 0.08 \
    --print_freq 40 --save_freq 20 --val_freq 1 \
    --use_track_poses \
    --output_dir output/combined5-3 \
    --random_pose 1 