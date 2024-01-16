.PHONY: data_download data_mismatch_label data_yolov8_pytorch_txt_format
	data_model_input data finetune_xlarge finetune_nano finetune_small
	finetune_medium finetune_large finetune_baseline finetune_all finetune_regions
	evaluate_all_on_common_validation_set evaluate_all_on_their_own_validation_set 
	evaluate_all hyperparameters_search all

setup: requirements.txt
	pip install -r requirements.txt

data_download:
	python src/data/download.py --to ./data/01_raw

data_mismatch_label:
	python src/data/mismatch_labels.py \
		--from data/09_external/label_mismatch/data.csv \
		--to data/04_feature/label_mismatch/data.csv

data_model_input:
	python src/data/yolov8/build_model_input.py \
		--to data/05_model_input/yolov8/ \
		--raw-root-rs-labelled data/01_raw/rs_storage_open/benthic_datasets/mask_labels/rs_labelled \
                --yolov8-pytorch-txt-format-root data/04_feature/yolov8/benthic_datasets/mask_labels/rs_labelled \
		--csv-label-mismatch-file data/04_feature/label_mismatch/data.csv \
		--loglevel info


data_yolov8_pytorch_txt_format:
	python src/data/yolov8/pytorch_txt_format.py \
		--from data/01_raw \
		--to data/04_feature

data: data_download data_mismatch_label data_yolov8_pytorch_txt_format data_model_input

finetune_baseline:
	python src/train/yolov8/cli.py \
		--experiment-name current_baseline \
		--epochs 5 \
		--model yolov8m-seg.pt \
		--data data/05_model_input/yolov8/v2/SEAFLOWER_BOLIVAR_and_SEAFLOWER_COURTOWN_and_SEAVIEW_ATL_and_SEAVIEW_IDN_PHL_and_SEAVIEW_PAC_AUS_and_TETES_PROVIDENCIA/data.yaml

finetune_regions:
	python src/train/yolov8/cli.py \
		--experiment-name current_best_xlarge \
		--epochs 100 \
		--model yolov8x-seg.pt \
		--imgsz 1024 \
		--close_mosaic 35 \
		--degrees 45 \
		--flipud 0.5 \
		--translate 0.2 \
		--data-list \
		data/05_model_input/yolov8/v2/SEAFLOWER_BOLIVAR/data.yaml \
		data/05_model_input/yolov8/v2/SEAFLOWER_COURTOWN/data.yaml \
		data/05_model_input/yolov8/v2/SEAVIEW_ATL/data.yaml \
		data/05_model_input/yolov8/v2/SEAVIEW_IDN_PHL/data.yaml \
		data/05_model_input/yolov8/v2/SEAVIEW_PAC_AUS/data.yaml \
		data/05_model_input/yolov8/v2/TETES_PROVIDENCIA/data.yaml \
		--loglevel info

finetune_nano:
	python src/train/yolov8/cli.py \
		--experiment-name current_best_nano \
		--epochs 100 \
		--model yolov8n-seg.pt \
		--imgsz 1024 \
		--close_mosaic 35 \
		--degrees 45 \
		--flipud 0.5 \
		--translate 0.2 \
		--data data/05_model_input/yolov8/v2/SEAFLOWER_BOLIVAR_and_SEAFLOWER_COURTOWN_and_SEAVIEW_ATL_and_SEAVIEW_IDN_PHL_and_SEAVIEW_PAC_AUS_and_TETES_PROVIDENCIA/data.yaml \
		--loglevel info

finetune_small:
	python src/train/yolov8/cli.py \
		--experiment-name current_best_small \
		--epochs 100 \
		--model yolov8s-seg.pt \
		--imgsz 1024 \
		--close_mosaic 35 \
		--degrees 45 \
		--flipud 0.5 \
		--translate 0.2 \
		--data data/05_model_input/yolov8/v2/SEAFLOWER_BOLIVAR_and_SEAFLOWER_COURTOWN_and_SEAVIEW_ATL_and_SEAVIEW_IDN_PHL_and_SEAVIEW_PAC_AUS_and_TETES_PROVIDENCIA/data.yaml \
		--loglevel info

finetune_medium:
	python src/train/yolov8/cli.py \
		--experiment-name current_best_medium \
		--epochs 100 \
		--model yolov8m-seg.pt \
		--imgsz 1024 \
		--close_mosaic 35 \
		--degrees 45 \
		--flipud 0.5 \
		--translate 0.2 \
		--data data/05_model_input/yolov8/v2/SEAFLOWER_BOLIVAR_and_SEAFLOWER_COURTOWN_and_SEAVIEW_ATL_and_SEAVIEW_IDN_PHL_and_SEAVIEW_PAC_AUS_and_TETES_PROVIDENCIA/data.yaml \
		--loglevel info

finetune_large:
	python src/train/yolov8/cli.py \
		--experiment-name current_best_large \
		--epochs 120 \
		--model yolov8l-seg.pt \
		--imgsz 1024 \
		--close_mosaic 35 \
		--degrees 45 \
		--flipud 0.5 \
		--translate 0.2 \
		--data data/05_model_input/yolov8/v2/SEAFLOWER_BOLIVAR_and_SEAFLOWER_COURTOWN_and_SEAVIEW_ATL_and_SEAVIEW_IDN_PHL_and_SEAVIEW_PAC_AUS_and_TETES_PROVIDENCIA/data.yaml \
		--loglevel info

finetune_xlarge:
	python src/train/yolov8/cli.py \
		--experiment-name current_best_xlarge \
		--epochs 140 \
		--model yolov8x-seg.pt \
		--imgsz 1024 \
		--close_mosaic 35 \
		--degrees 45 \
		--flipud 0.5 \
		--translate 0.2 \
		--data data/05_model_input/yolov8/v2/SEAFLOWER_BOLIVAR_and_SEAFLOWER_COURTOWN_and_SEAVIEW_ATL_and_SEAVIEW_IDN_PHL_and_SEAVIEW_PAC_AUS_and_TETES_PROVIDENCIA/data.yaml \
		--loglevel info

finetune_all: finetune_baseline finetune_regions finetune_xlarge finetune_large finetune_medium finetune_small finetune_nano

evaluate_all_on_their_own_validation_set:
	python src/evaluate/yolov8/cli.py \
	  --to data/08_reporting/yolov8/evaluation/ \
	  --model-root-path data/06_models/yolov8/segment/ \
	  --n-qualitative-samples 15 \
	  --batch-size 16 \
	  --num-workers 64 \
	  --random-seed 42 \
	  --loglevel info

evaluate_all_on_common_validation_set:
	python src/evaluate/yolov8/cli.py \
	  --to data/08_reporting/yolov8/evaluation/ \
	  --model-root-path data/06_models/yolov8/segment/ \
	  --data-root-path data/05_model_input/yolov8/v2/SEAFLOWER_BOLIVAR_and_SEAFLOWER_COURTOWN_and_SEAVIEW_ATL_and_SEAVIEW_IDN_PHL_and_SEAVIEW_PAC_AUS_and_TETES_PROVIDENCIA/ \
	  --n-qualitative-samples 15 \
	  --batch-size 16 \
	  --num-workers 64 \
	  --random-seed 42 \
	  --loglevel info

evaluate_all: evaluate_all_on_their_own_validation_set evaluate_all_on_common_validation_set

hyperparameters_search:
	python src/train/yolov8/hyperparameters_search.py

all: setup data finetune_all evaluate_all
