# Semantic-Preserving Adversarial Text Attacks

This repository contains Keras implementations of the paper: [Semantic-Preserving Adversarial Text Attacks](http://arxiv.org/abs/2108.10015).
<p align="center">
<img src="https://github.com/AdvAttack/BU-SPO/blob/main/image/fig-flowchart.png" width=100% height=100%>
</p>

## Requirements
* tensorflow == 1.15.2
* Keras == 2.2.4
* spacy == 2.1.4
* nltk == 3.4.5
* pandas == 0.23.4
* OpenHowNet == 0.0.1a8
* numpy == 1.15.4
* scikit_learn == 0.21.2
* If you did not download WordNet before, use `nltk.download('wordnet')` to do so.(Cancel the code comment on line 20 in `BU_SPO_paraphrase. py`) 


## Usage

* Download IMDB, AG's News and Yahoo! Answer datasets from [Google Drive](https://drive.google.com/drive/folders/1uvIYFvP21_YpAojuJ_UJ3CfWIq6DdDwr?usp=sharing) and place them in `/data_set`.
* Download `glove.6B.100d.txt`from [google drive](https://drive.google.com/file/d/1eUV5XW-B0CKRAyHsnp89cHc-s0psRot-/view?usp=sharing) and place the file in `/`.
* Use our pretrained model stored in `/runs` or train models by running `training.py`.
* Run `bigram.py` to generate bigram candidates or use the prelearnd bigram data in `/bigram`.
* To ensure the quick reproducibility, we provide HowNet candidate in [google drive](https://drive.google.com/drive/folders/18b_opVai9igJMze4h_Ip0wewuW2czuRi?usp=sharing). To recalculate the HowNet candidate set, run `build_embeddings.py`, `gen_pos_tag.py`, `lemma.py` and `gen_candidates.py` under the `/hownet_candidates` for each dataset.
* Run `BU_SPO_fool.py` to generate adversarial examples using BU-MHS.
* If you want to train or fool different models, reset the argument in `training.py`and`BU_SPO_fool.py`.


## Contact

Please contact Xinghao Yang or [Wei Liu](https://wei-research.github.io/) at firstname.lastname@uts.edu.au if you're interested to collaborate on this research!
