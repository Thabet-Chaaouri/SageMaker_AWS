# Exercice3_AWSSageMaker_HF
The customer wants a code example on how to fine-tune multilabel text classifier for content moderation fully on AWS.

This code example was built based on this [Hugging Face documentation about Amazon Sagemaker](https://huggingface.co/docs/sagemaker/index)

## First step: build the train.py taining script
it is a standard training script that:
* Uses Trainer API so it can scale easily into a distributed training afterwards.
* Receives hyperparameters and sagemaker environment variables as command line arguments, so add a part in the beginning to parse those arguments. 

For the purpose of this exercise, I used the tweets_hate_speech_detection dataset from Hugging Face hub. 
The labels for this dataset are:
* 0 (no hate speech)
* 1 (hate speech), which is a binary calssification problem. 

If you are working on a multilabel text classifier, it is the same code, you should only adjust the classifier head to your problem. you should construct two dictionnaries, one mapping the ids to labels and the other mapping labels to ids. below is an example for a dataset from HuggingFace hub:

```python
from transformers import AutoModelForSequenceClassification

label_names = train_dataset.feature.names

id2label = {i: label for i, label in enumerate(label_names)}
label2id = {v: k for k, v in id2label.items()}

model = AutoModelForSequenceClassification.from_pretrained(
    args.model_name,
    id2label=args.id2label,
    label2id=args.label2id,
)
```

checkout `train.py` script file.

## Second step: launch training on Amazon SageMaker

As you want a code example fully AWS, I chose to launch the training from a sagemaker notebook instance inside AWS.
This is one method. Other method is to launch the train from your local environement but the first one is easier and it avoids you dealing with networking and security problems connecting your OnPrem environemet to the cloud (Unless you have already a pipeline like that).

So create an aws sagemaker notebook instance:

* choose the least expensive instance that suffiecs for your task. In my region, it was `ml.t3.medium`.
* You should create an IAM sagemaker execution role and attach it to your instance. this roles gives the permission to your instance to call Sagemaker and access any S3 bucket that has sagemaker in the name.
* If you have your `train.py` script in a git repository, you can benefit from the facultatif option of cloning that repo to your notebook instance environment. then, using jupyter lab you can easily commit and push your work as you go.

![Capture](https://user-images.githubusercontent.com/87118784/221403751-37d23416-b0b3-4e8d-b1e5-abd16c8c76a2.PNG)

Once your instance is created, launch it, choose a notebook with a Pytorch kernel and start typing your code. 
Checkout my well commented `Fine_tuning_AWS.ipynb` notebook to walk you out through this process. In this notebook, thanks to Sagemaker and the Hugging Face Deep Learning Containers on AWS:
* First, I make a non-distributed code for finetuning a simple classifier
* Then I scale to a distributed setup taking adavantage of the Trainer API and the flexibility of the Estimator (The answer to question 2)

I launch my training. Once finished with one line of code I can deploy my model.

## Question 3: MLOps Workflow

