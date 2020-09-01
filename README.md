# Sentence Simplification with Deep Reinforcement Learning
This is an implmentation of the DRESS (**D**eep **RE**inforcement **S**entence **S**implification) model described in [Sentence Simplification with Deep Reinforcement Learning](http://aclweb.org/anthology/D/D17/D17-1062.pdf)


# Datasets
The *wikismall* and *wikilarge* datasets can be downloaded on [Github](https://github.com/louismartin/dress-data/raw/master/data-simplification.tar.bz2) or on [Google Drive](https://drive.google.com/open?id=0B6-YKFW-MnbOYWxUMTBEZ1FBam8).

8 references *wikilarge* test set can be downloaded here https://github.com/cocoxu/simplification/tree/master/data/turkcorpus

Copyright of the *newsela* dataset belongs to https://newsela.com. Please contact newsela.com to obtain the dataset https://newsela.com/data/

# System Output
If you are looking for system output and don't bother to install dependencies and train a model (or run a pre-trained model), the ``all-system-output`` folder is for you.

Additional system outputs can be found in the [EASSE text simplification library](https://github.com/feralvam/easse/tree/master/easse/resources/data/system_outputs). 

# Dependencies
* [CUDA 7.5](http://www.nvidia.com/object/cuda_home_new.html)
* [Torch](https://github.com/torch)
* [fb-python](https://github.com/facebook/fblualib/tree/master/fblualib/python)
* [pysari](https://github.com/XingxingZhang/pysari)

Note that this model is tested using an old version of torch (available [here](https://drive.google.com/open?id=0B6-YKFW-MnbOZ0gxNk56MjhQWjA))


# Train a Reinforcement Learning Simplification Model

## Step 1: Train an Encoder-Decoder Attention Model
```
CUDA_VISIBLE_DEVICES=$ID th train.lua --learnZ --useGPU \
    --model EncDecAWE \
    --attention dot \
    --seqLen 85 \
    --freqCut 4 \
    --nhid 256 \
    --nin 300 \
    --nlayers 2 \
    --dropout 0.2 \
    --lr $lr \
    --valid $valid \
    --test $test \
    --optimMethod Adam \
    --save $model \
    --train $train \
    --validout $validout --testout $testout \
    --batchSize 32 \
    --validBatchSize 32 \
    --maxEpoch 30 \
    --wordEmbedding $wembed \
    --embedOption fineTune \
    --fineTuneFactor 0 \
    | tee $log
```
Details see `experiments/wikilarge/encdeca/train.sh`. Note in `newsela` and `wikismall` datasets, you should use `--freqCut 3`. 

If you want to generate simplifications from a pre-trained Encoder-Decoder Attention model, use the following command:
```
CUDA_VISIBLE_DEVICES=3 th generate_pipeline.lua \
    --modelPath $model \
    --dataPath $data.test \
    --outPathRaw $output.test \
    --oriDataPath $oridata.test \
    --oriMapPath $orimap | tee $log.test
```
Details see `experiments/wikilarge/encdeca/generate/run_std.sh`.

## Step 2: Train a Language Model for the Fluency Reward
See details in `experiments/wikilarge/dress/train_lm.sh`

## Step 3: Train a Sequence Auto-Encoder for the Relevance Reward
Create dataset `scripts/get_auto_encoder_data/gen_data.sh`
See details in `experiments/wikilarge/dress/train_auto_encoder.sh`

## Step 4: Train a Reinforcement Learning Model
See details in `experiments/wikilarge/dress/train_dress.sh`. Run a pre-trained `DRESS` model using this script `experiments/wikilarge/dress/generate/dress/run_std.sh`.

## Step 5: Train a Lexical Simplification Model 
To train a lexical simplification model, you need to obtain soft word alignments in the training data, which are assigned by a pre-trained Encoder-Decoder Attention model. See details in `experiments/wikilarge/dress/run_align.sh`.

After you obtain the alignments, you can train a lexical simplification model using `experiments/wikilarge/dress/train_lexical_simp.sh`.

Lastly, you can apply the lexical simplification model with DRESS `experiments/wikilarge/dress/generate/dress-ls/run_std.sh`.

# Pre-trained Models
https://drive.google.com/open?id=0B6-YKFW-MnbOTVRMSURFbXYxNjg

# Evaluation
Please be careful about the automatic evaluation. <br>
You can use our released code and models to produce output for different models (i.e., EncDecA, Dress and Dress-Ls). But please make sure your evaluation settings follow the settings in our paper.

## BLEU
The evaluation pipeline accompanied in our code released produces single reference BLEU scores. 

#### WikiLarge
To be consistant with previous work, you should use 8 references wikilarge test set (availabel at https://github.com/cocoxu/simplification/tree/master/data/turkcorpus)

Therefore, to get the numbers on wikilarge, you should use scripts that support multi-bleu evalution (e.g., [joshua](https://github.com/cocoxu/simplification/#the-text-simplificaiton-system) or mtevalv13a.pl).

Checkout details for BLEU evaluation of wikilarge [here](https://github.com/XingxingZhang/dress/tree/master/experiments/evaluation/BLEU)

## FKGL
Make sure your FKGL is on corpus level.

## SARI
The evaluation pipeline accompanied in our code released produces sentence-level SARI scores. You can use this simplification system (available [here](https://github.com/cocoxu/simplification/#the-text-simplificaiton-system)) to produce corpus level SARI scores.

Checkout details for SARI evaluation [here](https://github.com/XingxingZhang/dress/tree/master/experiments/evaluation/SARI)


# Citation
```
@InProceedings{D17-1063,
  author = 	"Zhang, Xingxing
		and Lapata, Mirella",
  title = 	"Sentence Simplification with Deep Reinforcement Learning",
  booktitle = 	"Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing",
  year = 	"2017",
  publisher = 	"Association for Computational Linguistics",
  pages = 	"595--605",
  location = 	"Copenhagen, Denmark",
  url = 	"http://aclweb.org/anthology/D17-1063"
}
```


