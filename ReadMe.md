### Trained ELMO embeddings on SWB Speech Corpus from scratch


### Directory Structure 
src
|models 
    - checkpoint   - Download trained model checkpoint  
    - weights      - Downloads trained model weights        
    - options.json   - Downloads options.json with n_characters=262 for inference       
    Download link https://drive.google.com/drive/folders/1OmjWWBc1FMF7YDvQip_pDOTFvTiAAqsn?usp=sharing  

|swb     
    - swb/swb-train.csv  
    - swb/swb-test.csv  
    - swb/swb-train.csv 

|Report.pdf - Assignment report containing the euclidean and cosine distance


# How to run 

### Data prepration
``` python dataprep.py ```   

``` ls swb/train | wc -l ```

### Set options.json in models/checkpoint
{"bidirectional": true, "char_cnn": {"activation": "relu", "embedding": {"dim": 16}, "filters": [[1, 32], [2, 32], [3, 64], [4, 128], [5, 256], [6, 512], [7, 1024]], "max_characters_per_token": 50, "n_characters": 261, "n_highway": 2}, "dropout": 0.1, "lstm": {"cell_clip": 3, "dim": 4096, "n_layers": 2, "proj_clip": 3, "projection_dim": 512, "use_skip_connections": true}, "all_clip_norm_val": 10.0, "n_epochs": 10, "n_train_tokens": 1410521, "batch_size": 128, "n_tokens_vocab": 19558, "unroll_steps": 20, "n_negative_samples_batch": 128}

### Model Training

``` 
export CUDA_VISIBLE_DEVICES=""
cd bilm-tf-master
python train_elmo.py --train_prefix="../swb/train/*" --vocab_file "../swb/vocab.txt" --save_dir "../models/checkpoint" 
```

### Save model weights
``` python bin/dump_weights.py --save_dir '../models/checkpoint' --outfile '../models/weights/swb_weights.hdf5' ```

### Prediction and distance calculation
```
cd bilm-tf-master
mv -v bin/bilm/* bilm/
```

- Copy options.json and modify to
set : 'n_characters': 262   

- Run ``` python distance.py ```  
