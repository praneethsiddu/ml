# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 12:32:44 2021

@author: praneeth
"""
"""
Created by: praneeth partapu
"""

import streamlit as st



import pandas as pd 

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from lazypredict.Supervised import LazyClassifier, LazyRegressor
from sklearn.model_selection import train_test_split
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
st.set_option('deprecation.showPyplotGlobalUse', False)

    
def main():
 st.title("Machine Learning Automation")
 img = mpimg.imread('eda.jpg')
 st.image(img,use_column_width=True,caption='EDA') 
 st.sidebar.title("")
 data = st.sidebar.file_uploader("Upload Dataset", type=['csv'])
 activites = ["ExploringDataAnalysis","Pandas-Profiling","Data Visualization","LazyRegressor","LazyClassifier"]
 choice = st.sidebar.selectbox("Select Actvity", activites)
 
 if data is not None:
     @st.cache
     def load_csv():
         csv = pd.read_csv(data)
         return csv
     df = load_csv()
     df1=df.loc[:10000]
     st.success("Data File Uploaded Successfully")
 else:
     st.warning("Waiting for user to upload the cse file")
 
 if choice == 'ExploringDataAnalysis' and data is not None:
           
			st.subheader("Exploratory Data Analysis")
			# Data Show
			if st.checkbox("Show Data"):
				select_ = st.radio("HEAD OR TAIL",('All','HEAD','TAIL'))
				if select_ == 'All':
					st.dataframe(df)
				elif select_ == 'HEAD':
					st.dataframe(df.head())
				elif select_ == 'TAIL':
					st.dataframe(df.tail())
			# Columns
			if st.checkbox("Show Columns"):
				select_ = st.radio("Select Columns",('All Columns','Specific Column'))
				if select_ == "All Columns":
					st.write(df.columns)
				if select_ == "Specific Column":
					col_spe = st.multiselect("Select Columns To Show",df.columns)
					st.write(df[col_spe])

			# Show Dimension
			if st.checkbox("Show Dimension"):
				select_ = st.radio('Select Dimension',('All','Row','Column'))
				if select_ == "All":
					st.write(df.shape)
				elif select_ == "Row":
					st.write(df.shape[0])
				elif select_ == "Column":
					st.write(df.shape[1])

			# Summary of dataset
			if st.checkbox("Summary of Data Set"):
				st.write(df.describe())


			# Value Counts
			if st.checkbox("Value Count"):
				select_ = st.multiselect("Select values",df.columns.tolist())
				st.write(df[select_].count())
 
     

			# Show data Type
			if st.checkbox("Show Data Type"):
				select_ = st.radio("Select ",('All Columns','Specific Column'))
				if select_ == "All Columns":
					st.write(df.dtypes)
				elif select_ == "Specific Column":
					s = st.multiselect("Select value",df.columns.tolist())
					st.write(df[s].dtypes)
 elif choice=="Pandas-Profiling":

    if data is None:
        st.warning("No file Provided to work on")
    else:
        pr = ProfileReport(df1, explorative=True)
        st.header('**Input DataFrame**')
        st.write(df)
        st.write('---')
        st.header('**Pandas Profiling Report**')
        st_profile_report(pr)
     
     
                  
                    
       
 elif choice=="Data Visualization" and data is not None:
     st.write(df.shape)
     st.subheader("Data Visualization")
     if st.checkbox("Quick Analysis"):
         select_ = st.radio("Select Type for Quick Analysis",('Count Plot','Line chart','Bar chart','area chart','Scatter Plot','Correlation Heatmap','Histogram','Pair Plot'))
         if select_ == "Count Plot":
             st.write(df1.dtypes)
             s = st.selectbox('select the column',df1.columns)
             ax = sns.countplot(df1[s])
             ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
             st.write(sns.countplot(df1[s]))
             plt.tight_layout()
             st.pyplot()
         if select_ == "Line chart":
             st.write(df1.dtypes)
             s = st.multiselect("Select Columns To Show",df1.columns)
             st.line_chart(df1[s])
         if select_=="Bar chart":
             st.write(df1.dtypes)
             s = st.multiselect("Select Columns To Show",df1.columns)
             st.bar_chart(df1[s])
         if select_=="area chart":
             st.write(df1.dtypes)
             s = st.multiselect("Select Columns To Show",df1.columns)
             st.area_chart(df1[s])
         if select_ == 'Scatter Plot':
             st.write(df1.dtypes)
             x = st.selectbox('Select X Column',df1.columns)
             y = st.selectbox('Select Y Column',df1.columns)
             st.write(x,y)
             st.write(sns.scatterplot(x,y,data=df1))
             st.pyplot()   
         if select_=='Correlation Heatmap':
             st.write(sns.heatmap(df1.corr()))
             st.pyplot()
         if select_ == "Histogram":
             st.write(df1.dtypes)
             x = st.selectbox('Select Numerical Variables',df1.columns)
             st.write(sns.distplot(df1[x]))
             st.pyplot()
         if select_=="Pair Plot":
               st.write(sns.pairplot(df1))
               st.pyplot()
 elif choice== "LazyRegressor" and data is not None:
     df1=df.loc[:100] # FOR TESTING PURPOSE, COMMENT THIS OUT FOR PRODUCTION
     X = df1.iloc[:,:-1] # Using all column except for the last column as X
     Y = df1.iloc[:,-1]# Selecting the last column as Y
     st.markdown('**1.2. Dataset dimension**')
     st.write('X')
     st.info(X.shape)
     st.write('Y')
     st.info(Y.shape)
     st.markdown('**1.3. Variable details**:')
     st.write('X variable (first 20 are shown)')
     st.info(list(X.columns[:20]))
     st.write('Y variable')
     st.info(Y.name)
     split_size = st.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)
     X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size = split_size)
     models_train,predictions_train,models_test,predictions_test=LazyRegressordf( X_train, X_test, Y_train, Y_test)
     st.subheader('2. Table of Model Performance')
     st.write('Training set')
     st.write(predictions_train)
     st.write('Test set')
     st.write(predictions_test)
 elif choice=="LazyClassifier" and data is not None:
     df=df.loc[:100]
     X = df.iloc[:,:-1] # Using all column except for the last column as X
     Y = df.iloc[:,-1]# Selecting the last column as Y
     st.markdown('**1.2. Dataset dimension**')
     st.write('X')
     st.info(X.shape)
     st.write('Y')
     st.info(Y.shape)
     st.markdown('**1.3. Variable details**:')
     st.write('X variable (first 20 are shown)')
     st.info(list(X.columns[:20]))
     st.write('Y variable')
     st.info(Y.name)
     split_size = st.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)
     X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size = split_size)
     models_train,predictions_train,models_test,predictions_test=LazyClassifierdf(X_train, X_test, Y_train, Y_test)
     st.subheader('2. Table of Model Performance')
     st.write('Training set')
     st.write(predictions_train)
     st.write('Test set')
     st.write(predictions_test)
     
@st.cache      
def LazyRegressordf( X_train, X_test, Y_train, Y_test):
    reg = LazyRegressor(verbose=0,ignore_warnings=False, custom_metric=None)
    models_train,predictions_train = reg.fit(X_train, X_train, Y_train, Y_train)
    models_test,predictions_test = reg.fit(X_train, X_test, Y_train, Y_test)
    return models_train,predictions_train,models_test,predictions_test

     

@st.cache     
def LazyClassifierdf(X_train, X_test, Y_train, Y_test):
    
    reg = LazyClassifier(verbose=0,ignore_warnings=False, custom_metric=None)
    models_train,predictions_train = reg.fit(X_train, X_train, Y_train, Y_train)
    models_test,predictions_test = reg.fit(X_train, X_test, Y_train, Y_test)
    return models_train,predictions_train,models_test,predictions_test
    
     
             
                       
                
                         
					             
if __name__ == "__main__":
    main()
    
