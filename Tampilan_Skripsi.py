import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.dates as mdates
import pandas as pd 
from datetime import datetime, timedelta
import yfinance as yf
import streamlit as st  

st.title (" PERAMALAN HARGA SAHAM MENGGUNAKAN METODE ALGORITMA SUPPORT VECTOR REGRESSION (SVR)")
st.markdown("Studi Kasus : Saham PT. Kalbe Farma, TBK") 

date1, date2 = st.beta_columns(2)
date_def = "20150102"
start = date1.date_input('Start Date', datetime.strptime(date_def,'%Y%m%d')) 
end = date2.date_input('End Date')
forecast_date = f'{end + timedelta(days=1)}'
df = yf.download("KLBF.JK", start, end)
st.write("Melakukan proses peramalan tanggal ",forecast_date)
st.subheader("Data Harga Saham KLBF")
st.write(df)

#Download Data Optional
import base64
info1, info2, info3 = st.beta_columns(3)
download_df = base64.b64encode(df.to_csv(index=False).encode()).decode()
info1.markdown(
	f'<a href="data:file/csv;base64,{download_df}" download="data.csv">Download Data</a>',
	unsafe_allow_html=True
		)
info3.markdown('`  Sumber : Yahoo Finance  `')

if st.checkbox("Tampilkan Diagram Pergerakan Saham"):
	st.subheader("Diagram Pergerakan Saham")
	graph = st.line_chart(df["Close"])

df_data = df
df_data.reset_index(inplace=True,drop=False)
#st.write(df_data)	
x_all = pd.DataFrame(df_data.Date.astype(str).str.split('-').tolist(),columns="year month date".split())
def tranformasi_data(x_all, x_tr):
	scaler = StandardScaler()
	scaler.fit(x_all)
	x_all_tr = scaler.transform(x_tr)
	return x_all_tr
x_all_tr = tranformasi_data(x_all, x_all)

st.sidebar.title("Setting Parameter RBF")
Cd = st.sidebar.selectbox("C", [1, 10, 100, 1000], index=3)
gamma = st.sidebar.selectbox("Gamma", [0.1, 1, 10, 100], index=2)
cross_validation = st.sidebar.selectbox("Cross Validation", [3, 5, 10],index=2)

y_all=df_data.Close
def model(x_all_tr, y_all):
    gcs = GridSearchCV(SVR(kernel='rbf'),
                       param_grid={'C': [Cd], 'gamma': [gamma]},
                       cv = cross_validation,
                       refit=True,
                       return_train_score=False,
                       verbose=False,
                       n_jobs=-1,
                       scoring='neg_mean_squared_error')
    grid_result = gcs.fit(x_all_tr, y_all)
    best_params = grid_result.best_params_
    best_svr = SVR(kernel='rbf',C=best_params["C"], gamma=best_params["gamma"])
    best_svr.fit(x_all_tr, y_all)
    y_pred = best_svr.predict(x_all_tr)
    return y_pred, best_svr
y_pred, best_svr = model(x_all_tr, y_all)


st.subheader("Grafik Perbandingan Harga Saham KLBF dan Perhitungan SVR")
def plot(y_pred, df_data):
    fig, ax1 = plt.subplots(figsize=(25,15))
    plt.plot(df_data.Date, df_data.Close)
    monthyearFmt = mdates.DateFormatter('%Y')
    ax1.xaxis.set_major_formatter(monthyearFmt)
    _ = plt.xticks(rotation=90)
    plt.plot(df_data.Date, y_pred, c='r', label='RBF model')
    plt.scatter(df_data.Date, df_data.Close, c='b', label='Data')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Support Vector Regression')
    plt.legend(fontsize = '30')
    st.pyplot()
st.set_option('deprecation.showPyplotGlobalUse', False)
plot(y_pred, df_data)

#Linear Regression
x = x_all_tr
y = df_data.Close
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.3,random_state = 1)
reg = LinearRegression(n_jobs = -1).fit(x, y)
y_pred_lr = reg.predict(x_test)

plot_df = pd.DataFrame({'Actual':y_test,'Pred':y_pred_lr})

#Error Linear Regression
def rmse(y_test, y_pred_lr): 
    y_all_lr = np.array(y_test)
    return np.sqrt(np.square(np.subtract(y_test,y_pred_lr)).mean())

def mape(y_test, y_pred_lr): 
    y_all_lr = np.array(y_test)
    return np.mean(np.abs((y_test - y_pred_lr) / y_test)) * 100

RMSE_lr = 'RMSE : {0:.3f}'.format(rmse(y_test,y_pred_lr))
MAPE_lr = 'MAPE : {0:.3f}'.format(mape(y_test, y_pred_lr))


#Error SVR
def rmse(y_all, y_pred): 
    y_all = np.array(y_all)
    return np.sqrt(np.square(np.subtract(y_all,y_pred)).mean())

def mape(y_all, y_pred): 
    y_all = np.array(y_all)
    return np.mean(np.abs((y_all - y_pred) / y_all)) * 100

RMSE = 'RMSE : {0:.3f}'.format(rmse(y_all,y_pred))
MAPE = 'MAPE : {0:.3f}'.format(mape(y_all, y_pred))

#--------------
st.header("Tabel Hasil Perbandingan Metode Algoritma")
col1, col2 = st.beta_columns(2)
col1.subheader("SVR")
col1.write(RMSE)
col1.write(MAPE)
col2.subheader("Linear Regression")
col2.write(RMSE_lr)
col2.write(MAPE_lr)

#Prediksi 21 Mei 2021
prediksi_tanggal = '2021-04-21'
y, m, d = prediksi_tanggal.split('-')
c = [[y, m, d]]
c_tr = tranformasi_data(x_all, c)
hasil_prediksi = best_svr.predict(c_tr)
st.write("Menurut hasil dari perhitungan Support Vector Regression (SVR) dengan kernel RBF berikut Parameter ** C = %d **,** Gamma = %d **dan **CV = %d** yang telah diinput didapatkan hasil harga saham pada tanggal **%s** sebesar ** %d ** Rupiah"%(Cd, gamma, cross_validation, forecast_date, hasil_prediksi))
