<!--
<style>
img{
    width:80%;
    padding-left: 10%;
}
</style>
-->

# TextGT: A Double-View Graph Transformer on Text for Aspect-Based Sentiment Analysis

We introduce graph Transformer into text representation learning and apply it to the ABSA task. To tightly combine the 2 text learning processes in graph view and sequence view, we propose a novel double-view **g**raph **T**ransformer for **text** called **TextGT**. The overview of TextGT is as follows:<br/><br/><img src="resources/TextGT.png" style="width:80%; padding-left:10%" /> <br/> 


Additionally, we propose a new algorithm to implement graph convolutional modules which densely pass messages constructed with edge features, and one of such modules called **TextGINConv** is specifically employed as the graph-view operator in our TextGT. (We also implement TextGCNConv and TextGATConv, for simplicity we call the framework **TextGraphConv**) 


TextGT is built using [PyTorch](https://pytorch.org/) as well as [transformers](https://huggingface.co/docs/transformers/index), and TextGraphConv is based on sparse message passing graph convolutional modules from [PyG](https://www.pyg.org/). Specifically *Pytorch v1.12.1* and *transformers v4.26.1* are required, and package requirements are concluded in `requirements.txt`. Hardwares used are 2 NVIDIA GeForce RTX 3090 GPUs. 

```bash

# Python environment setup 
conda create -n text_gt python=3.9 
conda activate text_gt 
pip install -r requirements.txt 
```


### Priliminaries 

Due to the size limit of the sumpplementary material, the datasets and the GloVe file are not included here in this directory. 

The datasets can be downloaded from github repository of the baseline [SSEGCN](https://github.com/zhangzheng1997/SSEGCN-ABSA), [DualGCN](https://github.com/CCChenhao997/DualGCN-ABSA) or [CDT](https://github.com/Guangzidetiaoyue/CDT_ABSA). Datasets contained there have been already preprocessed by CoreNLP. If the users want to preprocess from scratch, then please refer to [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/simple.html) to download the util package named like `stanford-corenlp-full-***` and put it under this directory, and in Python start with `from stanfordcorenlp import StanfordCoreNLP` and `nlp = StanfordCoreNLP(r'./stanford-corenlp-full-***')` to use it. 

After downloading the datasets, to run TextGT w/o BERT, please download and unzip the GloVe file (`glove.840B.300d.zip`) from [https://nlp.stanford.edu/projects/glove/](https://nlp.stanford.edu/projects/glove/) and put it under `./glove`. Then use commands like `python ./prepare_vocab.py --data_dir ./dataset/Restaurants_corenlp --vocab_dir ./dataset/Restaurants_corenlp` to prepare the vocab for each dataset. As for the BERT version, the pre-trained model will be downloaded automatically while running. 


**The log files of some experimental runs are under directory `./logs`, corresponding to the results we report in the main paper or the technical appendix.** 



### Running TextGT 
```bash

# Please use environment command like "CUDA_VISIBLE_DEVICES=0" to specify the cuda device number 

# Restaurant w/o BERT
python train.py --model_name text-gt --num_layers 8 --scheduler linear --warmup 2 --ffn_dropout 0.4 --attn_dropout 0.2 --balance_loss 

# Restaurant w BERT 
python train.py --model_name text-gt-bert --ffn_dropout 0.5 --attn_dropout 0.2 --balance_loss 

# Laptop w/o BERT 
python train.py --model_name text-gt --num_layers 5 --ffn_dropout 0.4 --attn_dropout 0.1 --dataset laptop 

# Laptop w BERT 
python train.py --model_name text-gt-bert --num_layers 3 --ffn_dropout 0.5 --attn_dropout 0.2 --balance_loss --dataset laptop 

# Twitter w/o BERT 
python train.py --model_name text-gt --hidden_dim 50 --ffn_dropout 0.3 --attention_heads 1 --attn_dropout 0.1 --use_rnn --dataset twitter 

# Twitter w BERT 
python train.py --model_name text-gt-bert --num_layers 7 --ffn_dropout 0.5 --attn_dropout 0.1 --balance_loss --dataset twitter 

```


### Abalation Study 

We conduct ablation studies to demonstrate the effectiveness of each components making up TextGT. We replace Transformer layer with TextGINConv and the resulting model is called **TextGIN**. And **Transformer** means the graph conv is replaced by Transformer layer in TextGT. TextGT (GAT) and TextGT (GCN) mentioned in the main paper can actually realized by changing command arguments `graph_conv_type`. To run TextGIN and Transformer: 

```bash
python train.py --model_name text-gin # and other arguments for the specific datasets 
python train.py --model_name text-transformer # and other arguments for the specific datasets 
```

### Depth Study

To further show the superiority of TextGT, we vary model layers and observe test performance with respect to model depth. For varying depth of our TextGT, just change the command line argument `num_layers`. 

### Case Study 

To compare TextGT with the other SOTA methods in the ability to discriminate aspects, we conduct a case study. Use the scripts under `./case_study` to show the test cases and aspects corresponding to line numbers. As for generating the file of "target-prediction" pairs corresponding to line numbers, please uncomment `self._show_cases()` in `train.py` before running. 

### Visualization 
We further validate TextGT from the perspective of model interpretation. Use `--model_name text-gt-v` to train a TextGT which can output word attention matrices. Then change the path of the generated model state dict file in `output_words_attn.py`. The code of drawing procedure is in module `./visualization/visualize.py` which is called by `output_words_attn.py`. 
After that, run: 

```bash 
python output_words_attn.py --sentence_id [Number] # and other arguments for the specific datasets 
```

Two visualization examples are as follows: <br/><br/><img src="resources/words_attention_restaurant5.png" style="width:40%; padding-left:5%"/> <img src="resources/words_attention_laptop4.png" style="width:40%; padding-left:5%"/> 

---

# Appendix 


### Comparison to Other GT Constructing Ways 

We additionally perform a comparison experiment to show our TextGT constructed in an alternating way is surperior over other GTs in model architecture. Specifically, we compare to GNN+Transformer, Transformer+GNN, ParallelGT (Transformer layer and Graph Conv proceed in parallel and their outputs are fused later on) and TextTG (also an alternating model like ours, but the Transformer layer is before Graph Conv in each block). To run these models, please use `--model_name` to specify `gnn-transformer`, `transformer-gnn`, `parallel-gt` or `text-tg`. 

### Different Combinations of $n$ and $m$ 

To explore the influence of different numbers of Graph Convs and Transformer Layers ($n$ and $m$, respectively) in each TextGT block, we further perform configuration studies on Restaurant and Twitter, and the results are as follows. 
<br/><img src="resources/number_study_restaurant.png" style="width:40%; padding-left:5%"/><img src="resources/number_study_twitter.png" style="width:40%; padding-left:5%"/> 

Generally the model performance is degraded with the increase in $n$ and $m$, whilst $n=m=1$ is obviously the best configuration, which just corresponds to our TextGT. To reproduce the results, please specify the related model using `--model_name gt-n-m` with $n$ and $m$ replaced by 1, 2 or 4. 
