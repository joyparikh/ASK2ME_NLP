
import xlrd
import numpy as np
import re
from nltk.corpus import stopwords
import string
from nltk.stem import PorterStemmer
#from nltk.stem import LancasterStemmer
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import classification_report

def get_data():
    book      =   xlrd.open_workbook('DataSetToExportToCollboratorsMay22_2019.xlsx')
    dat_book  =   xlrd.open_workbook('CancerAKAForJoy.xlsx')
    sheet     =   book.sheet_by_index(0)
    sheet0    =   dat_book.sheet_by_index(0)
    sheet1    =   dat_book.sheet_by_index(1)
    sheet2    =   dat_book.sheet_by_index(2)
    title           =   sheet.col_slice(colx=1,end_rowx=6579,start_rowx=1)
    abstract        =   sheet.col_slice(colx=2,end_rowx=6579,start_rowx=1)
    am_penetrance   =   sheet.col_slice(colx=3,end_rowx=6579,start_rowx=1)    
    am_incidence    =   sheet.col_slice(colx=4,end_rowx=6579,start_rowx=1)
    penetrance      =   sheet.col_slice(colx=5,end_rowx=6579,start_rowx=1)
    incidence       =   sheet.col_slice(colx=6,end_rowx=6579,start_rowx=1)
    polymorphism    =   sheet.col_slice(colx=7,end_rowx=6579,start_rowx=1)
    germline        =   sheet.col_slice(colx=8,end_rowx=6579,start_rowx=1)
    somatic         =   sheet.col_slice(colx=9,end_rowx=6579,start_rowx=1)
    cancer_tokens   =   sheet0.col_slice(colx=0,end_rowx=750,start_rowx=1)
    gene_tokens     =   sheet1.col_slice(colx=0,end_rowx=830,start_rowx=1)
    syndrome_tokens =   sheet2.col_slice(colx=0,end_rowx=580,start_rowx=1)
    
    return title,abstract,am_penetrance,am_incidence,penetrance,incidence,polymorphism,germline,somatic,cancer_tokens,gene_tokens,syndrome_tokens

def data2val(title,abstract,am_penetrance,am_incidence,penetrance,incidence,polymorphism,germline,somatic,cancer_tokens,gene_tokens,syndrome_tokens):
    for n in np.arange(len(title)):
        title[n]           =   title[n].value
        abstract[n]        =   abstract[n].value
        am_penetrance[n]   =   am_penetrance[n].value
        penetrance[n]      =   penetrance[n].value
        am_incidence[n]    =   am_incidence[n].value
        incidence[n]       =   incidence[n].value
        polymorphism[n]    =   polymorphism[n].value
        germline[n]        =   germline[n].value
        somatic[n]         =   somatic[n].value
    
    for n in np.arange(len(cancer_tokens)):
        cancer_tokens[n]   =   cancer_tokens[n].value
        (cancer_tokens[n]).lower()
        
    for n in np.arange(len(gene_tokens)):
        gene_tokens[n]     =   gene_tokens[n].value
        gene_tokens[n].lower()
    
    for n in np.arange(len(syndrome_tokens)):
        syndrome_tokens[n] =   syndrome_tokens[n].value
        syndrome_tokens[n].lower()

    cancer_tokens          =   cancer_tokens[::-1]
    gene_tokens            =   gene_tokens[::-1]
    syndrome_tokens        =   syndrome_tokens[::-1]
    
    return title,abstract,am_penetrance,am_incidence,penetrance,incidence,polymorphism,germline,somatic,cancer_tokens,gene_tokens,syndrome_tokens

def text_proc(mess):
    port = PorterStemmer()
    # 1. removed punctuation
    nopunc    =    [c for c in mess if c not in string.punctuation]
    nopunc    =    ''.join(nopunc)
    nopunc    =    nopunc.lower()
    
    
    # 2. tokenisation
    for n in np.arange(len(cancer_tokens)):
        nopunc   =   nopunc.replace(cancer_tokens[n],"cancer")
        
    for n in np.arange(len(gene_tokens)):
        nopunc   =   nopunc.replace(gene_tokens[n],"gene")
        
    for n in np.arange(len(syndrome_tokens)):
        nopunc   =   nopunc.replace(syndrome_tokens[n],"gene")
        
        
    # 3. remove consecutive entries
    nopunc = re.sub(r'\b(.+)(\s+\1\b)+',r'\1',nopunc)
    
    
    # 4. removing stopwords
    nopunc = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    
    
    # 5. stemming
    for ind in np.arange(len(nopunc)):
        nopunc[ind] = port.stem(nopunc[ind])
    return nopunc 

def plot_learning_curve(pipeline, X, y, ylim=None, n_jobs=None, train_sizes=np.linspace(.1,1.0,5)):
    
    title       =     "LearningCurve" 
    cv          =      ShuffleSplit(n_splits=10,test_size=0.2, random_state=0)
    estimator   =      pipeline.named_steps['classifier'].best_estimator_
    vector_ip   =      pipeline.named_steps['tfidf'].transform(pipeline.named_steps['bow'].transform(X))
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, vector_ip,y,cv=cv, n_jobs=n_jobs, train_sizes = train_sizes)
    train_scores_mean = np.mean(train_scores,axis=1)
    train_scores_std = np.std(train_scores,axis=1)
    test_scores_mean = np.mean(test_scores,axis=1)
    test_scores_std = np.std(test_scores,axis=1)
    
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean-train_scores_std,train_scores_mean+train_scores_std,alpha=0.1,color='r')
    plt.fill_between(train_sizes, test_scores_mean-test_scores_std,test_scores_mean+test_scores_std,alpha=0.1,color='g')
    plt.plot(train_sizes, train_scores_mean, 'o-',color='r',label="Training Score")
    plt.plot(train_sizes, test_scores_mean, 'o-',color='g',label="CV Score")
    
    plt.legend(loc="best")
    return plt

def task_one(am_penetrance,am_incidence,title,somatic,germline):
    indicesp     =     [i for i in np.arange(len(am_penetrance)) if am_penetrance[i]==1]
    indicesi     =     [i for i in np.arange(len(am_incidence)) if am_incidence[i]==1]
    indices      =     np.union1d(indicesp,indicesi)
    title1       =     np.delete(title,indices)
    label1       =     np.delete(somatic,indices)
    label2       =     np.delete(germline,indices)
    return title1,label1,label2,indices

def task_two(label2,title1,polymorphism,indices):
    indices2    =   [i for i in np.arange(len(label2)) if label2[i]==0]
    title2      =   np.delete(title1,indices2)
    labelt      =   np.delete(polymorphism,indices)
    labelft     =   np.delete(labelt,indices2)
    labelf      =   labelft.astype(float)

    return title2,labelf,indices2

def task_three(labelf,title2,indices,indices2,incidence,penetrance):
    indices3     =     [i for i in np.arange(len(labelf)) if labelf[i]==1.0]
    title3       =     np.delete(title2,indices3)
    label_i_1    =     np.delete(incidence,indices)
    label_p_1    =     np.delete(penetrance,indices)
    label_i_2    =     np.delete(label_i_1,indices2)
    label_p_2    =     np.delete(label_p_1,indices2)
    label_i      =     np.delete(label_i_2,indices3)
    label_p      =     np.delete(label_p_2,indices3)

    return title3,label_i,label_p,indices3

def metrics(title_te,label_te,pipeline):
    label_pred   =   pipeline.predict(title_te)
    print(classification_report(label_pred,label_te))
    return

def predict_final(test,pipeline_s,pipeline_g,pipeline_p,pipeline_pe,pipeline_i):
    som_pred     =   pipeline_s.predict(test)
    germ_pred    =   pipeline_g.predict(test)
    
    
    print("somatic :")
    print(som_pred)
    print("\n")
    
    print("germline :")
    print(germ_pred)
    print("\n")
    
    if germ_pred==1:
        poly_pred    =   pipeline_p.predict(test)
        print("polymorphism :")
        print(poly_pred)
        print("\n")
        
        if poly_pred==0:
            inc_pred     =   pipeline_i.predict(test)
            pen_pred     =   pipeline_pe.predict(test)
            print("penetrance :")
            print(pen_pred)
            print("\n")
            
            print("incidence :")
            print(inc_pred)
            print("\n")
            
        else:
            print("No further classification as polymorphism is present")
            
    else:
        print("No further classification as germline is absent")

def clf_train(title,label):
    param_gd     =     {'C':[1,10,100],'gamma':[0.1,1,0.01]} 
    grid         =     GridSearchCV(SVC(),param_gd,verbose=3)
    
    pipeline = Pipeline([
        ('bow',CountVectorizer(analyzer=text_proc)),
        ('tfidf',TfidfTransformer()),
        ('classifier',grid)
    ])

    title_tr,  title_te,  label_tr,  label_te   =   train_test_split(title,label,test_size=0.1)
    pipeline.fit(title_tr,label_tr)
    return pipeline,title_te,label_te
    
if __name__ == '__main__':

    print("getting data...")
    title,abstract,am_penetrance,am_incidence,penetrance,incidence,polymorphism,germline,somatic,cancer_tokens,gene_tokens,syndrome_tokens = get_data()
    title,abstract,am_penetrance,am_incidence,penetrance,incidence,polymorphism,germline,somatic,cancer_tokens,gene_tokens,syndrome_tokens = data2val(title,abstract,am_penetrance,am_incidence,penetrance,incidence,polymorphism,germline,somatic,cancer_tokens,gene_tokens,syndrome_tokens)

    for n in np.arange(len(title)):
        title[n]=title[n]+' '+abstract[n]
        
    #con = False;
    print("======================")
    title1,label1,label2,indices = task_one(am_penetrance,am_incidence,title,somatic,germline)
    print("Successfully executed task1, enter y to see removed entries")
    ch = input("enter any other character to continue:")
    if ch=='y':
        print("removed entries")
        print(indices)
    print("\n-x-x-x-x-x-x-x-x-x-x\n")

    print("training classifier for somatic")
    pipeline_s,title_te_s,label_te_s = clf_train(title1[1:50],label1[1:50])
    print("training classifier for germline")
    pipeline_g,title_te_g,label_te_g = clf_train(title1[1:50],label2[1:50])
    print("training done")
    print("getting metrics for somatic...")
    metrics(title_te_s,label_te_s,pipeline_s)
    print("getting metrics for germline...")
    metrics(title_te_g,label_te_g,pipeline_g)
    print("learning curve somatic")
    plot_learning_curve(pipeline_s,title1[1:500],label1[1:500],ylim=(0.7,1.01),n_jobs=4)
    plt.show()
    print("learning curve germline")
    plot_learning_curve(pipeline_g,title1[1:500],label2[1:500],ylim=(0.7,1.01),n_jobs=4)
    plt.show()
    
    print("======================")
    title2,labelf,indices2 = task_two(label2,title1,polymorphism,indices)
    print("Successfully executed task2, enter y to see removed entries")
    ch = input("enter any other character to continue:")
    if ch=='y':
        print("removed entries")
        print(indices2)
    print("\n-x-x-x-x-x-x-x-x-x-x\n")
    
    print("training classifier for polymorphism")
    pipeline_p,title_te_p,label_te_p = clf_train(title2[1:500],labelf[1:500])
    print("training done")
    print("getting metrics for polymorphism")
    metrics(title_te_p,label_te_p,pipeline_p)
    print("learning curve polymorphism")
    plot_learning_curve(pipeline_p,title2[1:500],labelf[1:500],ylim=(0.7,1.01),n_jobs=4)
    plt.show()
    
    print("======================")
    title3,label_i,label_p,indices3 = task_three(labelf,title2,indices,indices2,incidence,penetrance)
    print("Successfully executed task3, enter y to see removed entries")
    ch = input("enter any other character to continue:")
    if ch=='y':
        print("removed entries")
        print(indices3)
    print("\n-x-x-x-x-x-x-x-x-x-x\n")
    
    print("training classifier for penetrance")
    pipeline_pe,title_te_pe,label_te_pe = clf_train(title3[1:50],label_p[1:50])
    print("training classifier for incidence")
    pipeline_i,title_te_i,label_te_i = clf_train(title3[1:50],label_i[1:50])
    print("training done")
    print("getting metrics for penetrance...")
    metrics(title_te_pe,label_te_pe,pipeline_pe)
    print("getting metrics for incidence...")
    metrics(title_te_i,label_te_i,pipeline_i)
    print("learning curve penetrance")
    plt = plot_learning_curve(pipeline_pe,title3[1:500],label_p[1:500],ylim=(0.7,1.01),n_jobs=4)
    plt.show()
    print("learning curve incidence")
    plt = plot_learning_curve(pipeline_i,title3[1:500],label_i[1:500],ylim=(0.7,1.01),n_jobs=4)
    plt.show()

    print("testing the trained classifiers on random entry : ")
    ch = input("Enter any indice inbetween 1 to 6578")
    predict_final([title[int(ch)]],pipeline_s,pipeline_g,pipeline_p,pipeline_pe,pipeline_i)
