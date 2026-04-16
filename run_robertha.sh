DEVICE="cuda:1"
epochs=3

for seed in 23 42 77 112 135; do
	MODEL_DIR="models/seed$seed"
	LOG_DIR="logs/seed$seed"
	mkdir -p $MODEL_DIR $LOG_DIR

	#GLUE (9), SuperGLUE (4), and PAWS (3)
	for ds in  cola sst2 mrpc qqp stsb mnli qnli wnli rte boolq copa wic wsc paws pawsx_en pawsx_de; do
		python robertha.py \
			--regularization esr \
			--task $ds \
			--train True \
			--device $DEVICE \
			--epochs $epochs \
			--model_dir $MODEL_DIR \
			--seed $seed \
			--noise_type absolute	| tee $LOG_DIR/robertha_$ds.log
	done

	#AdvGLUE (6)
	for ds in adv_sst2 adv_qqp adv_mnli adv_qnli adv_rte adv_mnli_mismatched; do
		python robertha.py \
			--regularization esr \
			--task $ds \
			--model_dir $MODEL_DIR \
			--seed $seed \
			--train False	| tee $LOG_DIR/robertha_$ds.log
	done
done
