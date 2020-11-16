# AniMaker: AI-Generated story concepts from Anime plot synopses </br>

Created by **Kelsey Glenn** </br>

[GitHub](www.github.com/kelseyfglenn) • [LinkedIn](www.github.com/linkedin.com/in/kfglenn) </br>

## Overview </br>
This project was designed to help automate the process of story concept creation by training a neural language model to generate plot synopses. The generator is built by fine-tuning OpenAI’s GPT-2 language model on roughly 14,000 Anime plot synopses.

I use Anime synopsis data from MyAnimeList to train the model in hopes that the heavy use of story tropes and otherwise “predictable” nature of Anime story premises will make them well-suited to coherent generation.

I additionally create a regression model to predict MyAnimeList user scores as a proxy for story quality. The model uses Random Forest regression on NMF-generated topic-document matrices. After semi-randomly generating a large number of texts, I apply the model to assist in “curating” those of high quality.

Finally, code for a small deployment via Steamlit is included which allows users to generate text from their own starting seed as well as utilize the score prediction model.

A brief warning: this project includes data containing potentially explicit material and may contain inappropriate or sensitive content.

## Project Directory</br>
**Modules**</br>
<ul>
<li><b>clean_generations.py</b> - function for cleaning generator output file</li>
<li><b>clean_text.py</b> - function for cleaning training data</li>
<li><b>generator.py</b> - functions for custom synopsis generation</li> 
<li><b>score_text.py</b> - function for using score prediction model on a synopsis</li>
</ul>

**Notebooks**</br>
<ul>
<li><b>finetuning_and_generation.ipynb</b> - GPT-2 fine-tuning and initial generation</li>
<li><b>store_and_score.ipynb</b> - bulk synopsis generation and scoring</li>
<li><b>gpt-2-medium-pytorch.ipynb</b> - **deprecated**, scrapped GPT-2 Medium model training w/ PyTorch</li>
<li><b>hyperparameter_tuning.ipynb</b> - tuning score prediction model hyperparameters</li>
<li><b>score_prediction.ipynb</b>- experiments and optimization of score prediction model</li>
</ul>

**Training Data**</br>
<ul>
<li><b>Anime.csv</b> - initial dataset containing synopses, scores and other data, courtesy of [Adrian L. Ludosan](https://www.kaggle.com/aludosan/myanimelist-anime-dataset-as-20190204).</li>
<li><b>gpt_training_data.csv</b> - cleaned training texts stored as a single column CSV for GPT-2</li>
</ul>




