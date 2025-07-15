# SED with text

## Code repository for "Sound event detection with audio-text models and heterogeneous temporal annotations"

how to use:
1. edit paths `settings.yaml`
2. run `prepare_data.py` to create the mels & sound event data
3. run `prepare_synthetic_captions.py` to tokenize training captions
4. run `prepare_splits.py` to create the train/val/test splits
5. run `prepare_train_split_probs.py` to collect the reference data for data resampling (if using)
6. run `train_multimodal.py` to train the model
7. evaluate using `eval_multimodal.py` for overall results and `eval_tag.py` for erroneous tag injections
