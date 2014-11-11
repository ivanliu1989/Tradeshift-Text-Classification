Tradeshift-Text-Classification
==============================

In the late 90's, Yann LeCun's team pioneered the successful application of machine learning to optical character recognition. 25 years later, machine learning continues to be an invaluable tool for text processing downstream from the OCR process.

Tradeshift has created a dataset with thousands of documents, representing millions of words. In each document, several bounding boxes containing text are selected. For each piece of text, many features are extracted and certain labels are assigned.

In this competition, participants are asked to create and open source an algorithm that correctly predicts the probability that a piece of text belongs to a given class.

<b>ps. This solution is based on an 8GB RAM machine. Due to the limits of machine, the solution can definitely be improved further. </b>

### What I've learnt from this competition ###
1. Ensemble models are really powerful and easy ways to win the price.
2. XGBoost is an efficient and powerful tool in gradient boosting.
3. Online algorithms such as Vowpal Wabbit can passby the limits of machine to some extent and they are good at dealing with large data set. 
4. Feature engineering like interaction, one-hotted and hash are necessary before training models. 
5. RAM plays key role in such competition (categorical features, hash), 16GB seems to be minimum requirements to achieve top 10% in the leaderboard. 
6. Next challenge, VAZU-CTR, would be another perfect competition to implement all tips above. 
