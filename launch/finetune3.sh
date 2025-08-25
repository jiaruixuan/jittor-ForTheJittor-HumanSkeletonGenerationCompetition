export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1


mpirun -np 2 python train_combined5.py \
    --train_data_list data/train_list.txt --val_data_list data/val_list.txt \
    --feat_dim 256 --num_joints 52 \
    --num_samples 2048 --vertex_samples 1536 \
    --pretrained_model output/combined5-2/best_competition_model.pkl \
    --epochs 80 --learning_rate 2e-5 \
    --optimizer adam --use_track_poses \
    --output_dir output/combined5-2-ft \
    --random_pose 1 \
    --skeleton_loss_weight 1.0 --skin_loss_weight 0.5 \
    --skel_alpha 0.3 --skel_beta 0.8 --bone_len_weight 0.02 \