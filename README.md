# Recommender System 2023 Challenge - Polimi

<p align="center">
  <img width="100%" src="https://i.imgur.com/tm9mSuM.png" alt="header" />
</p>
<p align="center">
    <img src="https://i.imgur.com/mPb3Qbd.gif" width="180" alt="Politecnico di Milano"/>
</p>

This repo contains the code and the data used in the [Polimi's Recsys Challenge 2023](https://www.kaggle.com/competitions/recommender-system-2023-challenge-polimi)
<br>
The goal of the competition was to create a recommender system for books, providing 10 recommendations for each user.

## Results

* Ranked 1st
* MAP@10 - private: &nbsp;0.13956
* MAP@10 - public: &nbsp;&nbsp;&nbsp;0.13977

## Goal
The application domain is book recommendation. 
The datasets we provide contains interactions of users with books, in particular, if the user attributed to the book a rating of at least 4.
The main goal of the competition is to discover which items (books) a user will interact with.

## Data description
The datasets includes around 600k interactions, 13k users, 22k items (books).
The training-test split is done via random holdout, 80% training, 20% test.
The goal is to recommend a list of 10 potentially relevant items for each user. MAP@10 is used for evaluation. You can use any kind of recommender algorithm you wish written in Python. 

## Evaluation
The evaluation metric for this competition is MAP@10.

## Recommender
Our best recommender is a hybrid composed of
* SLIM ElasticNet
* RP3Beta
* IALS
* ItemKNN

## Team
* [Jacopo Piazzalunga](https://github.com/Jacopopiazza)
* [Davide Salonico](https://github.com/???)

## Credits
This repository is based on [Maurizio Dacrema's repository](https://github.com/MaurizioFD/RecSys_Course_AT_PoliMi)
