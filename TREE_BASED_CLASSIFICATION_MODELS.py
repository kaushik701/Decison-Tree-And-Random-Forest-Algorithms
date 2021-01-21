#%%
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as image
import pandas as pd
import os
import warnings
from yellowbrick.target import ClassBalance
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz # display the tree within a Jupyter notebook
from IPython.display import SVG
from graphviz import Source
from IPython.display import display
from ipywidgets import interactive, IntSlider, FloatSlider, interact
import ipywidgets
from IPython.display import Image
from subprocess import call
from yellowbrick.model_selection import FeatureImportances
from yellowbrick.classifier import ROCAUC
from sklearn.linear_model import LogisticRegression
%matplotlib inline
plt.style.use('ggplot')
warnings.simplefilter('ignore')
# %%
plt.rcParams['figure.figsize'] = (12,8)
hr = pd.read_csv('data/employee_data.csv')
hr_orig = hr
hr.head()
# %%
hr.info()
# %%
pd.crosstab(hr.salary,hr.quit).plot(kind='bar')
plt.title('TurnOver Frequency on Salary Bracket')
plt.xlabel('Salary')
plt.ylabel('Frequency of TurnOver')
plt.show()
# %%
pd.crosstab(hr.department,hr.quit).plot(kind='bar')
plt.title('TurnOver Frequency for Department')
plt.xlabel('Department')
plt.ylabel('Frequency of TurnOver')
plt.show()
# %%
hr.drop(columns=['department','salary'],axis=1,inplace=True)
# %%
visualizer = ClassBalance(labels=['stayed','quit'])
visualizer.fit(hr.quit)
visualizer.show()
# %%
X = hr.loc[:,hr.columns!='quit']
y = hr.quit
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0,stratify=y)
# %%
@interact
def plot_tree(crit=['gini','entropy'],
              split=['best','random'],
              depth = IntSlider(min=1,max=30,value=2,continous_update=False),
              min_split=IntSlider(min=2,max=5,value=2,continous_update=False),
              min_leaf=IntSlider(min=1,max=5,value=1,continous_update=False)):
    estimator = DecisionTreeClassifier(random_state=0,criterion=crit,splitter=split,max_depth=depth,min_samples_split=min_split,min_samples_leaf=min_leaf)
    estimator.fit(X_train,y_train)
    print(accuracy_score(y_train,estimator.predict(X_train)))
    print(accuracy_score(y_test,estimator.predict(X_test)))
    graph = Source(tree.export_graphviz(estimator,out_file=None,feature_names=X_train.columns,class_names=['0','1'],filled=True))
    display(Image(data=graph.pipe(format='png')))
    return estimator
# %%
@interact
def plot_tree(crit=['gini','entropy'],
              bootstrap=['True','False'],
              depth = IntSlider(min=1,max=30,value=2,continous_update=False),
              forests = IntSlider(min=1,max=200,value=100,continous_update=False),
              min_split=IntSlider(min=2,max=5,value=2,continous_update=False),
              min_leaf=IntSlider(min=1,max=5,value=1,continous_update=False)):
    estimator = RandomForestClassifier(random_state=1,criterion=crit,bootstrap=bootstrap,max_depth=depth,min_samples_split=min_split,min_samples_leaf=min_leaf,n_jobs=-1,verbose=False)
    estimator.fit(X_train,y_train)
    print(accuracy_score(y_train,estimator.predict(X_train)))
    print(accuracy_score(y_test,estimator.predict(X_test)))
    num_tree = estimator.estimators_[0]
    graph = Source(tree.export_graphviz(num_tree,out_file=None,feature_names=X_train.columns,class_names=['0','1'],filled=True))
    display(Image(data=graph.pipe(format='png')))
    return estimator
# %%
plt.rcParams['figure.figsize'] = (12,8)
plt.style.use("ggplot")
rf = RandomForestClassifier(bootstrap='True',class_weight=None,criterion='gini',max_depth=3
                            ,max_features='auto',max_leaf_nodes=None,min_impurity_decrease=1
                            ,min_samples_split=2,min_weight_fraction_leaf=0.0,n_estimators=100
                            ,n_jobs=1,oob_score=False,random_state=1,verbose=False,warm_start=False)
viz = FeatureImportances(rf)
viz.fit(X_train,y_train)
viz.show()
# %%
visualizer = ROCAUC(rf,classes=['stayed','quit'])
visualizer.fit(X_train,y_train)
visualizer.score(X_test,y_test)
visualizer.poof()
# %%
dt = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=2,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=0,
            splitter='best')

visualizer = ROCAUC(dt, classes=["stayed", "quit"])

visualizer.fit(X_train, y_train)        # Fit the training data to the visualizer
visualizer.score(X_test, y_test)        # Evaluate the model on the test data
visualizer.poof();
# %%
from sklearn.linear_model import LogisticRegressionCV
logit = LogisticRegressionCV(random_state=1, n_jobs=-1,max_iter=500,
                             cv=10)

lr = logit.fit(X_train, y_train)

print('Logistic Regression Accuracy: {:.3f}'.format(accuracy_score(y_test, lr.predict(X_test))))

visualizer = ROCAUC(lr, classes=["stayed", "quit"])

visualizer.fit(X_train, y_train)        # Fit the training data to the visualizer
visualizer.score(X_test, y_test)        # Evaluate the model on the test data
visualizer.poof();
# %%
