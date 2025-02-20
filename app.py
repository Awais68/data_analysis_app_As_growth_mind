
import streamlit as st
import pandas as pd
import numpy as np

import seaborn as sns
from plotly import graph_objs as go

# add title
st.title('Data Analysis Application')
st.subheader('This is a simple data analysis application creade By AS Developers')

# load the data
# Dropdown list to choose a dataset
dataset_name = st.selectbox('Select a dataset', ['iris', 'titanic', 'tips', 'diamonds',])

# Load the selected dataset
df = sns.load_dataset(dataset_name)

# Option to upload own dataset
uploaded_file = st.file_uploader("Or upload your own dataset (CSV file)", type=["csv", "xlsx"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file) # assuming the uploaded file is in CSV format
# display the number of rows and Columns from teh selected dataset
st.write('Number of rows:', df.shape[0])
st.write('Number of columns:', df.shape[1])

# display the dataset
# st.write(df.head())
st.write(df)




#  display the columns names of selected data with their data types
st.write('Columns Name and Data types:', df.dtypes)

# print the null values if those are  > 0
if df.isnull().sum().sum() > 0:
    st.write('Null values in the dataset:', df.isnull().sum().sort_values(ascending=False))
else:
    st.write('No Null values in the dataset')

# display the summary statistics of the selected data
st.write('Summary Statistics:', df.describe())

# create a pairplot
st.subheader('Pairplot')
# select the column to be used as hue in pairplot (method 1)
hue_column = st.selectbox('Select a column as hue', df.columns)
st.pyplot(sns.pairplot(df, hue=hue_column))
# st.pyplot(sns.pairplot(df, hue='species', markers='o'))


# (method 2)
# select the specific column for X and Y axis from the dataset and the also selet the plot type
# x_axis = st.selectbox('Select the X-axis for the plot', df.columns)
# y_axis = st.selectbox('Select the Y-axis for the plot', df.columns)
# plot_type = st.selectbox('Select the plot type', ['scatter', 'line', 'bar', 'hist', 'box', 'kde'])

# # plot the data 
# if plot_type == 'line':
#     st.line_chart(df[[x_axis, y_axis]])
# elif plot_type == 'scatter':
#     st.scatter_chart([x_axis, y_axis])
# elif plot_type == 'bar':
#     st.bar_chart([ x_axis, y_axis])
# elif plot_type == 'hist':
#     df[x_axis].plot(kind='hist')
#     st.pyplot()
# elif plot_type == 'box':
#     df[x_axis].plot(kind='box')
#     st.pyplot()
# else:
#     df[[x_axis, y_axis]].plot(kind='kde')
#     st.pyplot()


# create a heatmap
st.subheader('Heatmap')
# select the columns which are numeric and then create a corr_matrix
numeric_columns = df.select_dtypes(include=[np.number]).columns
corr_matrix = df[numeric_columns].corr()

from plotly import graph_objs as go

# convert the seaborn heatmap plot to a plotly figure
heatmap_fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='Viridis'
    ))
    

st.plotly_chart(heatmap_fig)
# st.pyplot(sns.heatmap(df.corr_matrix(), annot=True, cmap='coolwarm'))
