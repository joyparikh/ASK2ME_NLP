# ASK2ME_NLP
There is a commented and directive jupyter notebook in this repository.

The initial parts are data processing i.e it is converted into a form which is usable by our classification models. Then we move onto function for data-preprocessing and a learning curve function. 

The steps followed for classification are as follows:
1. The ambiguous entries are removed from the dataset and the remaining entries are used for training and testing a somatic and a germline classifier.

2. The non-germline entries are removed and the remaining dataset is used for training and testing a polymorphism classifier.

3. the polymorphism entries are removed and the remaining dataset is used for training and testing a penetrance and an incidence classifier.

For every classifier, a grid and a pipeline is used. Gridsearch is used to find out the tunable parameters which provide the best classification result. Pipeline is used so that we do not have to deal with bag-of-words and TF-IDF vectors, rather we'll only pass the string abstracts and see the result.

For the various experimentation in data-preprocessing the text_proc function can be referred to. Whenever a certain part is not required it can be commented out. The implemented things are:

1. Removal of stopwords and punctuations
2. Tokenisation and removal of repetetive tokens
3. Stemming

1. Stopwords such as and,the,is are not very useful in classification purposes and hence were removed. Punctuation was also removed as it was not very useful as a distinguishing feature.

2. A dataset was provided where certain strings were to be replaced by a single token 'gene', 'syndrome' or 'cancer'. Further as an experiment the syndrome token was replaced by the gene token and consecutive entries of gene token were replaced by a single entry gene.

3. Stemming and Lammetisation are 2 techniques which are used to reduce the words to their basic form. For example running is reduced to run and catches is reduced to catch. The prefixes and suffixes are removed. Both were implemented and stemming was found in general to have a better result and was hence continued in the pre-processing. 
