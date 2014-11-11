Tradeshift-Text-Classification
==============================

In the late 90's, Yann LeCun's team pioneered the successful application of machine learning to optical character recognition. 25 years later, machine learning continues to be an invaluable tool for text processing downstream from the OCR process.

Tradeshift has created a dataset with thousands of documents, representing millions of words. In each document, several bounding boxes containing text are selected. For each piece of text, many features are extracted and certain labels are assigned.

In this competition, participants are asked to create and open source an algorithm that correctly predicts the probability that a piece of text belongs to a given class.

<b>ps. This solution is based on an 8GB RAM machine. Due to the limits of machine, the solution can definitely be improved further. </b>

### What I've learnt from this competition ###
1. Ensemble models are really powerful and easy ways to win the price.
2. XGBoost is a efficient and powerful tool in gradient boosting.
3. Online algorithms such as Vowpal Wabbit are helpful in dealing with large data set. 
4. Feature engineering like interaction, one-hotted and hash can help improve classifiers a lot. 
5. Next challenge, VAZU-CTR, would be another perfect competition to implement all tips above. 
