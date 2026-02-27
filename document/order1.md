  - --test_opt_physics
  - --test_opt_backend（默认 linearized_ktc）
  - --test_opt_steps（默认 20）
  - --test_opt_lr（默认 1e-2）
  - --test_opt_save_curve


python src/generate_simdata_32.py --output_dir data32 --num_train 40000 --num_valid 5000 --workers 8

python src/pack_dataset_to_h5.py --input_dir data32 --format h5  

python main.py inference --method cnn --config src/configs/cnn_config.yaml --checkpoint results/cnn_01/best_model.pth --dataset test2023 --sample_idx 2 --test_opt_physics --test_opt_mode contour_step --test_opt_steps 1000 --test_opt_stage2_steps 60

python main.py inference --method fno --config src/configs/fno_config.yaml --checkpoint results/fno_00/best_model.pth --dataset test2023 --sample_idx 1 --test_opt_physics --test_opt_mode contour_step --test_opt_steps 200 --test_opt_stage2_steps 30

python main.py train --method cnn --config src/configs/cnn_config.yaml --data_dir data32 --num_epochs 200 --batch_size 64 --result_dir cnn_sim32

python main.py train --method cnn --config src/configs/cnn_config.yaml --data_dir data32 --measurement_format matrix32 --num_epochs 200 --result_dir cnn32

python -m py_compile src/utils/visualization.py

python main.py inference --method fno --config src/configs/fno_config.yaml --checkpoint results/fno32_00/best_model.pth --dataset ktc_full --ktc_level 1 --measurement_format matrix32

python main.py inference --method cnn --config src/configs/cnn_config.yaml --checkpoint results/cnn32_00/best_model.pth --dataset ktc_eval --ktc_level 1 --measurement_format matrix32

python main.py inference --method cnn --config src/configs/cnn_config.yaml --checkpoint results/cnn32_01/best_model.pth --dataset ktc_full --ktc_level 1 --measurement_format matrix32

python main.py train --method cnn --config src/configs/cnn_config.yaml --data_dir data32 --measurement_format matrix32 --num_epochs 100 --result_dir cnn32

python main.py train --method cnn --config src/configs/cnn_config.yaml --data_dir data32 --measurement_format matrix32 --num_epochs 200 --resume results/cnn32_04/last.pth

python main.py inference --method cnn --config src/configs/cnn_config.yaml --checkpoint results/cnn32_04/best_model.pth --dataset ktc_full --ktc_level 1 --measurement_format matrix32

python main.py train --method cnn --config src/configs/cnn_config.yaml --data_dir data32 --measurement_format matrix32 --num_epochs 200 --result_dir cnn32

python main.py inference --method cnn --config src/configs/cnn_config.yaml --checkpoint results/cnn32_05/best_model.pth --dataset ktc_full --ktc_level 1 --measurement_format matrix32

python src/generate_simdata_32.py --output_dir data32_from_data --source_data_dir data --source_splits train,valid,test --workers 8 --overwrite

python main.py train --method cnn --config src/configs/cnn_config.yaml --data_dir data32_from_data --measurement_format matrix32 --num_epochs 200 --result_dir cnn32_from_data
