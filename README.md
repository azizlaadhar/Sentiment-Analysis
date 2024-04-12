[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/Jb7AnP5E)
## **Assignment Description**
- In the first part of this assignment, you will need to implement training (finetuning) and evaluation of a pre-trained language model ([RoBERTa](https://huggingface.co/docs/transformers/model_doc/roberta)) on a **Sentiment Analysis (SA)** task, which aims to determine whether a product review's emotional tone is positive or negative.

- For part-2, following the first finetuning task, you will need to identify the shortcuts (i.e. some salient or toxic features) that the model learnt for the specific task.

- For part-3, you are supposed to annotate 80 randomly assigned new datapoints as ground-truth labels. Additionally, the cross annotation should be conducted by another one or two annotators, and you will learn about how to calculate the agreement statistics as a significant characteristic reflecting the quality of a collected dataset.

- For part-4, since the human annotation is quite time- and effort-consuming, there are plenty of ways to get silver-labels from automatic labeling to augment the dataset scale, e.g., paraphrasing each text input in different words without changing its meaning. You will use a [T5](https://huggingface.co/docs/transformers/en/model_doc/t5) paraphrase model to expand the training data of sentiment analysis, and evaluate the improvement of data augmentation.

For Parts 1 and Part 2, you will need to complete the code in the corresponding `.py` files (`sa.py` for Part 1, `shortcut.py` for Part 2). You will be provided with the function descriptions and detailed instructions about the code snippet you need to write.


### Table of Contents
- **PART 1: Sentiment Analysis (33 pts)**
    - 1.1 Dataset Processing (10 pts)
    - 1.2 Model Training and Evaluation (18 pts)
    - 1.3 Fine-Grained Validation (5 pts)
- **PART 2: Identify Model Shortcuts (22 pts)**
    - 2.1 N-gram Pattern Extraction (6 pts)
    - 2.2 Distill Potentially Useful Patterns (8 pts)
    - 2.3 Case Study (8 pts)
- **PART 3: Annotate New Data (25 pts)**
    - 3.1 Write an Annotation Guideline (5 pts)
    - 3.2 Annotate Your Datapoints with Partner(s) (8 pts)
    - 3.3 Agreement Measure (12 pts)
- **PART 4: Data Augmentation (20 pts)**
    - 4.1 Data Augmentation with Paraphrasing (15 pts)
    - 4.2 Retrain RoBERTa Model with Data Augmentation (5 pts)
    
### Deliverables

- ✅ This jupyter notebook: `assignment2.ipynb`
- ✅ `sa.py` and `shortcut.py` file
- ✅ Checkpoints for RoBERTa models finetuned on original and augmented SA training data (Part 1 and Part 4), including:
    - `models/lr1e-05-warmup0.3/`
    - `models/lr2e-05-warmup0.3/`
    - `models/augmented/lr1e-05-warmup0.3/`
- ✅ Model prediction results on each domain data (Part 1.3 Fine-Grained Validation): `predictions/`
- ✅ Cross-annotated new SA data (Part 3), including:
    - `data/<your_assigned_dataset_id>-<your_sciper_number>.jsonl`
    - `data/<your_assigned_dataset_id>-<your_partner_sciper_number>.jsonl`
    - (for group of 3) `data/<your_assigned_dataset_id>-<your_second_partner_sciper_number>.jsonl`
- ✅ Paraphrase-augmented SA training data (Part 4), including:
    - `data/augmented_train_sa.jsonl`
- ✅ `tensorboard` directory with logs for all trained/finetuned models, including:
    - `tensorboard/part1_lr1e-05/`
    - `tensorboard/part1_lr2e-05/`
    - `tensorboard/part4_lr1e-05/`

### How to implement this assignment

Please read carefully the following points. All the information on how to read, implement and submit your assignment is explained in details below:

1. For this assignment, you will need to implement and fill in the missing code snippets for both the **Jupyter Notebook `assignment2.ipynb`** and the **`sa.py`**, **`shortcut.py`** python files.

2. Along with above files, you need to additionally upload model files under the **`models/`** dir, regarding the following models:
    - finetuned RoBERTa models on original SA training data (PART 1)  
    - finetuned RoBERTa model on augmented SA training data (PART 4)
  
3. You also need to upload model prediction results in Part 1.3 Fine-Grained Validation, saved in **`predictions/`**.

4. You also need to upload new data files under the **`data/`** dir (along with our already provided data), including:
    - new SA data with your and your partner's annotations (Part 3)
    - paraphrase-augmented SA training data (Part 4)

5. Finally, you will need to log your training using Tensorboard. Please follow the instructions in the `README.md` of the **``tensorboard/``** directory.

**Note**: Large files such as model checkpoints and logs should be pushed to the repository with Git LFS. You may also find that training the models on a GPU can speed up the process, we recommend using Colab's free GPU service for this. A tutorial on how to use Git LFS and Colab can be found [here](https://github.com/epfl-nlp/cs-552-modern-nlp/blob/main/Exercises/tutorials.md).
