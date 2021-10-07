# -------------------------------- IMPORTS -----------------------#
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from matplotlib import pyplot as plt
from PIL import Image
from sklearn.metrics import accuracy_score

# -------------------------------- IMAGE --------------------------#

img = Image.open('red_wine.jpg')
st.image(img)

# ------------------------------- INTERFACE -----------------------#
st.write("""
# Wine Quality Prediction App

###### This App predicts the Wine quality Good or bad! 
###### If quality of wine is greater than 6 than wine quality is Good or else its bad. Given below is the data recorded for wine testing.
    
""")
# -------------------------------- GETTING VALUES OF PARAMETER --------------------#
st.sidebar.header("Prediction Parameters")

def user_input_features():
    fixed_acidity = st.sidebar.slider('Fixed Acidity', 4.0, 16.0, 7.3)
    volatile_acidity = st.sidebar.slider('Volatile Acidity', 0.0, 2.0, 0.68)
    citric_acid = st.sidebar.slider('Citric Acid', 0, 2, 0)
    residual_sugar = st.sidebar.slider('Residual Sugar', 0.0, 20.0, 1.5)
    chlorides = st.sidebar.slider('Chlorides', 0.0, 1.0, 0.0627)
    free_sulfur_dioxide = st.sidebar.slider('Free Sulfur Dioxide', 1, 80, 17)
    total_sulfur_dioxide = st.sidebar.slider('Total Sulfur Dioxide', 1, 300, 20)
    density = st.sidebar.slider('Density', 0.0, 1.0, 0.998)
    pH = st.sidebar.slider('pH Level', 1.0, 4.0, 3.42)
    sulphates = st.sidebar.slider('Sulphates', 0.0, 2.0, 0.5)
    alcohol = st.sidebar.slider('Alcohol', 5, 17, 10)
    data_x = {
        'fixed acidity': fixed_acidity,
        'volatile acidity': volatile_acidity,
        'citric acid': citric_acid,
        'residual sugar': residual_sugar,
        'chlorides': chlorides,
        'free sulfur dioxide': free_sulfur_dioxide,
        'total sulfur dioxide': total_sulfur_dioxide,
        'density': density,
        'pH': pH,
        'sulphates': sulphates,
        'alcohol': alcohol,

    }
    features = pd.DataFrame(data_x, index=[0])
    return features


df = user_input_features()

# ----------------------------------------- DATA PRE-PROCESSING -------------------------#
st.subheader('Original Data Set')
with st.echo():

    data = pd.read_csv("winequality-red.csv")
    st.write(data)


st.subheader('Data values to be predicted')
st.write(df)

st.header('Data Preprocessing')
st.write(f' ##### Good Quality Wine:  1 \n ##### Bad Quality Wine:  0\n')
st.subheader('Introducing new column "Group Quality" which is classified ')
with st.echo():

    data["Group Quality"] = np.where(data["quality"] > 6, 1, 0)
    st.write(data[["quality", "Group Quality"]])


st.subheader('Total Wine Counts')
with st.echo():
    st.write(data["Group Quality"].value_counts())

st.subheader('Data sorted by group quality')
with st.echo():

    data_mod = data.sort_values(by="Group Quality", ascending=False)
    st.write(data_mod)


# ----------------------------------------------- LINEAR DISCRIMINANT ANALYSIS --------------------------------#
X = data_mod[data_mod.columns[0:11]]
Y = data_mod[data_mod.columns[12:]]



st.header('Linear Discriminant Analysis')
with st.echo():
    clf = LinearDiscriminantAnalysis(solver="eigen").fit(X, Y)
    st.subheader('Features')
    st.write(X)

st.subheader('Global Means')
with st.echo():
    global_mean = np.mean(X, axis=0)

    st.write(global_mean)


with st.echo():
    x_split = np.split(X, [0, 218])
    x_1 = x_split[1]
    x_2 = x_split[2]
st.subheader('Mean for Good quality')
with st.echo():
    mean_x_1 = np.mean(x_1, axis=0)
    st.write(mean_x_1)
st.subheader('Mean for Bad quality')
with st.echo():
    mean_x_2 = np.mean(x_2, axis=0)
    st.write(mean_x_2)
    mean = [mean_x_1, mean_x_2]


st.subheader('Mean corrected data for Good quality wine')
with st.echo():
    mean_corrected_1 = x_1 - global_mean
    st.write(mean_corrected_1)


st.subheader('Mean corrected data for Bad quality wine')
with st.echo():
    mean_corrected_2 = x_2 - global_mean
    st.write(mean_corrected_2)





st.subheader('Covariance for Good quality wine')
with st.echo():

    n_1 = len(mean_corrected_1)
    n_2 = len(mean_corrected_2)
    cov_1 = (mean_corrected_1.T.dot(mean_corrected_1)) / n_1
    st.write(cov_1)


st.subheader('Covariance for Bad quality wine')
with st.echo():

    cov_2 = (mean_corrected_2.T.dot(mean_corrected_2)) / n_2
    st.write(cov_2)



st.subheader('Pooled Covariance')
with st.echo():
    total_obs = n_1 + n_2
    cov_f = ((n_1 / total_obs) * cov_1) + ((n_2 / total_obs) * cov_2)
    st.write(cov_f)




cov_inv = np.linalg.inv(cov_f)


# ------------------------------------------------ PROBABILITIES ----------------------------------------#
st.subheader('Probabilities of both groups')
with st.echo():

    prob_1 = n_1 / total_obs
    prob_2 = n_2 / total_obs
    col1,col2 = st.columns(2)
    col1.metric(label=" Probability Group 1", value=prob_1)
    col2.metric(label=" Probability Group 2", value=prob_2)
    prob = [prob_1, prob_2]




# ---------------------------------------------- DISCRIMINANT FUNCTION ---------------------------------------#
st.subheader('Discriminant Functions')
with st.echo():
    x_pred = np.array(df)
    for x in range(0, 2):
        dis_f = ((mean[x].dot(cov_inv)).dot(x_pred.T)) - ((0.5 * mean[x]).dot(cov_inv)).dot(mean[x].T) + np.log(prob[x])
        st.write(f"The value for f{x + 1} :  {dis_f}")



# --------------------------------------------- GROUP DECISION ----------------------------------------------#
st.subheader('Prediction')
with st.echo():
    decision = clf.predict(df)
    st.write(f"The predicted value will belong to group  :  {clf.predict(df)}")

# --------------------------------------------- RESULT DISPLAY -----------------------------------#
if decision == [1]:
    st.write("""
    ## The Wine is of Good Quality!! 🍷
    """)
else:
    st.write("""
    ## The Wine is of Bad Quality !!🤮
    """)
# ----------------------------------------- ACCURACY ---------------------------------#
with st.echo():
    y_pred_all = clf.predict(X)
    acc = accuracy_score(Y, y_pred_all)
    st.write(f''
             f'#### The Accuracy Of Model is {acc * 100}')
# ---------------------------------------- VISUALISATION of 3 Features -----------------------------#
with st.echo():
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    a = ax.scatter(data_mod['sulphates'],data_mod['density'],data_mod['alcohol'], c=data_mod['Group Quality'])
    plt.legend(loc="best")
    st.write(fig)




