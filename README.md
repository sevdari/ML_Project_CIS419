# CIS 4190 Applied Machine Learning Project: Group 045
For this project, we were interested in studying sentiment prediction in NLP. Sentiment analysis is an important tool for organizations and businesses, as they seek to understand large amounts of text data. 

We were primarily interested in seeing how sentiments towards food were reflected in review data. For this project, we used the [Amazon Fine Foods Dataset](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews), taking in the review texts as the raw inputs to our model and trying to predict whether or not the reviews were overall positive or negative.

A complete description of the project can be found [here](https://docs.google.com/document/d/1ItTVHn8Xer-YdEf4eEcemQIk__AekbLh/edit?usp=sharing&ouid=111301309263338542076&rtpof=true&sd=true). 

Instructions on how to run each file:

## lstm.ipynb
The following two files should be available in the same directory:
- Reviews.csv: This file can be downloaded from the Kaggle link above.
- glove.840B.300d.txt: It can be downloaded [here](https://nlp.stanford.edu/projects/glove/). This file provides us with pretrained glove word vectors that have been trained on Common Crawl data, a snapshot of the whole web. 

The trainig process for this file was done using an EC2 instance from AWS. Apart from that, the other code cells should run in under a few minutes in most laptops. 

The best performance achieved on the validation set (which contained an equal number of samples from each class) was close to 90%.

Some examples of sentences and their classification:

<img width="585" alt="image" src="https://user-images.githubusercontent.com/88673859/234456738-90690405-8983-4203-9f91-99cc66dec552.png">

Snapshot of the EC2 Training:

<img width="314" alt="image" src="https://user-images.githubusercontent.com/88673859/234476454-9dadd4d6-662c-4204-9128-8029ee736ac0.png">
