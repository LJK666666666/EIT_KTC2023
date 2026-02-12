  - --test_opt_physics
  - --test_opt_backend（默认 linearized_ktc）
  - --test_opt_steps（默认 20）
  - --test_opt_lr（默认 1e-2）
  - --test_opt_save_curve


python src/generate_simdata_32.py --output_dir data32 --num_train 46080 --num_valid 5760 

python src/pack_dataset_to_h5.py --input_dir data32 --format h5  