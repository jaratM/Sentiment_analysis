# Analysing twitter data about specific topics using NLP techniques

In this work, we used state-of-the-art methods in NLP and processing techniques to understand the patterns and characteristics of tweets and predict the sentiment they carry. Specifically, we built a model that can classify a tweet as either positive or negative based on the sentiments it reverts. We decided to use two classes in order to accommodate the complexity of the problem and its consistent with ongoing research and applications in the field. 
Using our sentiment predictor, we also built an interactive visualization tool to help businesses interpret and visualize public sentiments for their product and brands. This tool enables the user to not only visualize plain sentiment distribution over the entire dataset, but also equips user to conduct sentiment analysis over the dimension of time and location.

## Prerequisites
for the notebook and the application both requisites are documented on the files head.

## Project structure
First we started by test many models therefor we used a notebook file "sentiment_analysis_platform.ipynb" where we processed tweets, encoded words to vectors using different words embedding techniques and tested many models and RNN based architectures. After we found the performant model we built a web application using Flask for deployement and plotly for ploting figures.
- App.py: contains the model to deploy.
- templates/index.html : contains the front end code built using html, bootstrap, JavaScript and plotly.

## Running the project
to run this project you first need to run the notebook file and persist the best model in our case is LSTM using BERT embeddings then add them to this project along with the tokenizer. After adding these files to your project run the commands below on your terminal:

1- cd to sentiment_analysis directory 

2- python3 app.py

By default, flask will run on port 5000.

3 - Navigate to URL http://localhost:5000

4 - fill in the form and if everything goes well, four figures must be on your screen.
