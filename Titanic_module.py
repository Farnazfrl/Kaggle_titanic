#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import pickle

class titanic_model():
    
    def __init__(self, model_file):
            
            with open('model_xgboost_titanic','rb') as model_file:
                self.model = pickle.load(model_file)
                self.train=pd.read_csv('train_titanic.csv')


    def load_and_preprocess(self,data_file):
        df_test=pd.read_csv(data_file,index_col='PassengerId')
        df=pd.concat([self.train,df_test],axis=0)
        
        df=df[[ 'Name', 'Sex','Age', 'Pclass','SibSp', 'Parch', 'Ticket',
           'Fare', 'Cabin', 'Embarked','Survived']]
        df['Cabin']=df['Cabin'].str[0]
        df['Ticket']=df['Ticket'].str.extract('(\d+)')
        df['Sex']=df['Sex'].map({'male':1,'female':0})
        df['Cabin'].loc[df['Cabin']=='T']=df['Cabin'].mode()[0]
        self.clean=df.copy()
        df['Title']=df['Name'].str.extract(' ([A-Z,a-z]+)\.', expand=False)
        df['Age'].fillna(df.groupby('Title')['Age'].transform('mean'),inplace=True)
        df['Embarked'].fillna(df['Embarked'].mode()[0],inplace=True)

        #impute cabins for passengers who have the same ticket number 
        idx=df.duplicated(subset='Ticket',keep=False)[df.duplicated(subset='Ticket',keep=False)].index.values
        df_same_ticket=pd.DataFrame()
        df_same_ticket=df.iloc[idx-1,:]
        same_ticket=df_same_ticket['Ticket'].unique()
        for ticket in same_ticket:
            if df_same_ticket.loc[df_same_ticket['Ticket']==ticket]['Cabin'].notnull().any():
                df_same_ticket['Cabin'].fillna(df_same_ticket.loc[df_same_ticket['Ticket']==ticket]['Cabin'].mode()[0])
        df['Cabin'].iloc[idx-1]=df_same_ticket['Cabin']

         #impute the remaining cabins       
        df['Cabin'].loc[df['Pclass']==1]=df['Cabin'].loc[df['Pclass']==1].fillna('Unk1')
        df['Cabin'].loc[df['Pclass']==2]=df['Cabin'].loc[df['Pclass']==2].fillna('Unk2')
        df['Cabin'].loc[df['Pclass']==3]=df['Cabin'].loc[df['Pclass']==3].fillna('Unk3')
        self.impute=df.copy()
        df_test=df.iloc[df_test.index-1,:]
        self.preproccesed_data=df.copy()
        self.test=df_test
        return df_test
    
    def create_feature_and_scale(self):
        df=self.preproccesed_data.copy()

        #creating a feature for sum of number of family members for each passenger
        df['Group']=df['Parch']+df['SibSp']

        #creating a binary feature for title of passengers
        df['Title'].loc[df['Title']=='Ms']='Miss'
        List_ordinary=['Mr','Miss','Mrs']
        for i in List_ordinary:
            df['Title'].loc[df['Title']==i]='Ordinary'
        List_special=['Rev','Dr','Col','Mlle','Major','Lady','Sir','Mme','Don','Capt','Countess','Jonkheer','Dona','Master']
        for i in List_special:
            df['Title'].loc[df['Title']==i]='Special'

        #converting the ordinaries who have specials in family to special
        idx=df.duplicated(subset='Ticket',keep=False)[df.duplicated(subset='Ticket',keep=False)].index.values
        df_same_Ticket=pd.DataFrame()
        df_same_Ticket=df.iloc[idx-1,:]
        for i in df_same_Ticket['Ticket'].unique():
            if  (df_same_Ticket['Title'].loc[df_same_Ticket['Ticket']==i]=='Special').any():
                df_same_Ticket['Title'].loc[df_same_Ticket['Ticket']==i]='Special'
        df['Title'].iloc[idx-1]=df_same_Ticket['Title']
        df['Title']=df['Title'].map({'Ordinary':0,'Special':1})


        #a numerical feature for mean of fare in each class
        df['Mean Fare by Class']=df.groupby('Pclass')['Fare'].transform('mean')

        #creating a numerical feature for sum of Titled, high Pclassed and young people  
        df['Titled/High/Young']=0
        for i in range(df.shape[0]):
            if (df['Title'].iloc[i]==1) and (df['Age'].iloc[i]<=df['Age'].mean()) and  (df['Fare'].iloc[i]>df['Fare'].loc[df['Pclass']==1].mean()):
                df['Titled/High/Young'].iloc[i]=3
            else:
                if ( ((df['Title'].iloc[i]==1) or  (df['Age'].iloc[i]<=df['Age'].mean())) and  (df['Fare'].iloc[i]>df['Fare'].loc[df['Pclass']==1].mean()) ) or ( ((df['Title'].iloc[i]=='Special') or (df['Fare'].iloc[i]>df['Fare'].loc[df['Pclass']==1].mean())) and (df['Age'].iloc[i]<=df['Age'].mean()) ) or ( ((df['Age'].iloc[i]<=df['Age'].mean()) or  (df['Fare'].iloc[i]>df['Fare'].loc[df['Pclass']==1].mean())) and (df['Title'].iloc[i]=='Special')  ):
                     df['Titled/High/Young'].iloc[i]=2
                else:

                    if (df['Title'].iloc[i]==1) or (df['Age'].iloc[i]<=df['Age'].mean()) or (df['Fare'].iloc[i]>df['Fare'].loc[df['Pclass']==1].mean()):
                        df['Titled/High/Young'].iloc[i]=1

        df['Title']=df['Name'].str.extract(' ([A-Z,a-z]+)\.', expand=False)
        df['Title'].loc[df['Title']=='Ms']='Miss'
        df['Title'].loc[df['Title']=='Dona']='Don'
        for i in List_ordinary:
            df['Title'].loc[df['Title']==i]='Ordinary'
        List_Special_family=['Mlle','Lady','Sir','Mme','Jonkheer']
        List_Special_position=['Rev','Dr','Col','Major','Don','Capt','Countess','Master']
        for i in List_Special_family:
            df['Title'].loc[df['Title']==i]='Special_family'
        for i in List_Special_position:
            df['Title'].loc[df['Title']==i]='Special_position'

        df.drop(['Name','Ticket','SibSp','Parch','Survived'],axis=1,inplace=True) 


        for col in ['Fare','Age','Group','Titled/High/Young','Mean Fare by Class']:
            df[col]=(df[col]-df[col].mean())/df[col].std()
        
        df[['Embarked','Cabin','Title','Pclass']]=df[['Embarked','Cabin','Title','Pclass']].astype('object')
        df_dummy=pd.get_dummies(df[['Embarked','Cabin','Title','Pclass']],drop_first=True)
        df=df.join(df_dummy)
        df.drop(['Embarked','Cabin','Title','Pclass'],axis=1,inplace=True)   
        df_test=df.iloc[self.test.index-1,:]
        self.create_features=df_test
        return df_test
    
    def predict(self):
        self.predicted_proba=self.model.predict_proba(self.create_features)
        self.results=results=np.where(self.predicted_proba[:,1]>=.45,1,0)
        results=pd.DataFrame(index=self.test.index)
        results['Probability']=self.predicted_proba[:,1]
        results['Predicton']=self.results
        return results

    
    
    
    
    


# In[ ]:




