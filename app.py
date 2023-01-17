from flask import Flask, request,render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from datetime import date
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense, LSTM

current_year = date.today().year
next_year = current_year + 1

cvt_eng = {'ข้าวนาปี' : 'NAPEE', 'ข้าวนาปรัง' : 'NAPRUNG', 'ข้าวโพดเลี้ยงสัตว์' : 'CORN', 'อ้อยโรงงาน' : 'AOI'}
cvt_mth = {'มกราคม' : 1, 'กุมภาพันธ์' : 2, 'มีนาคม' : 3, 'เมษายน' : 4, 'พฤษภาคม' : 5, 'มิถุนายน' : 6, 'กรกฎาคม' : 7, 'สิงหาคม' : 8, 'กันยายน' : 9, 'ตุลาคม' : 10, 'พฤศจิกายน' : 11, 'ธันวาคม' : 12}
cvt_kg = {'ข้าวนาปี' : 1, 'ข้าวนาปรัง' : 1, 'ข้าวโพดเลี้ยงสัตว์' : 1000, 'อ้อยโรงงาน' : 1}

def add_comma(number):
    return ("{:,}".format(number))

def create_dataset(dataset,target):
    st = dataset.index[0]
    xy = dataset.shift(1).merge(dataset[target], how = 'inner', on = 'Date')
    xy = xy.dropna()
    xy.index = pd.date_range(st, periods=len(dataset)-1, freq="M")
    xy = xy.drop(columns=[target+'_x'])
    return np.array(xy.iloc[:,:-1]), np.array(xy.iloc[:,-1])


def AOP_LR(prd,prv,mth):
  data = pd.read_csv("amt_of_prd_per_area/all_for_"+ cvt_eng[prd].lower()+".csv")
  data.index = data['year']
  del data['year']
  data = data[data.province == prv]
  data_corr = data[['humid_AM', 'humid_PM','Amount_of_product_per_area']]
  corr = data_corr.corr()['Amount_of_product_per_area']
  exogenous_features =  list(corr[(abs(corr)>=0.8) & (corr.index != 'Amount_of_product_per_area')].index)
  df = data[exogenous_features +  ['temp_AM', 'rain_AM', 'humid_AM', 'temp_PM', 'rain_PM', 'Amount_of_product_per_area']]

  # normalize the dataset
  scaler = MinMaxScaler(feature_range=(0, 1))
  df['Amount_of_product_per_area']= scaler.fit_transform(np.array(df['Amount_of_product_per_area']).reshape(-1,1))
  reg = LinearRegression().fit(df.drop(columns = ['Amount_of_product_per_area']), df['Amount_of_product_per_area'])

  # Predict
  fore_var = pd.read_csv("For_Forecast/all_forecast_aop.csv")
  pred_var = fore_var[(fore_var['province'] == prv) & (fore_var['year'] == current_year)]
  pred_var = pred_var[exogenous_features +  ['temp_AM', 'rain_AM', 'humid_AM', 'temp_PM', 'rain_PM']]
  predict_aop = scaler.inverse_transform(reg.predict(pred_var).reshape(-1,1))
  return predict_aop[0][0]

def PRICE_LSTM(prd,prv,mth):
  file_prd = "price/ALL_FOR_" + cvt_eng[prd] + "_PRICE.csv"
  data = pd.read_csv(file_prd)
  data['year'] = data['year'] -543
  data.year = data.year.astype(str)
  data.month = data.month.astype(str)
  data['Date'] = data['year'] + '-' + data['month']
  data.index = pd.to_datetime(data.Date, format='%Y-%m')
  del data['Date']
  data = data[data['price'] != 0]
  corr = data.corr(method="spearman")['price']
  exogenous_features = list(corr[(abs(corr)>=0.2) & (corr.index != 'price')].index)
  df = data[exogenous_features + ['price']]
  # normalize the dataset
  scaler = MinMaxScaler(feature_range=(0, 1))
  df['price']= scaler.fit_transform(np.array(df['price']).reshape(-1,1))
  # train test split
  train = df[df.index < (data['year'].max()+"-01-01")]
  test = df[df.index >= (data['year'].max()+"-01-01")]
  # tansform dataset
  trainX, trainY = create_dataset(train, 'price')
  testX, testY = create_dataset(test, 'price')
  # reshape input to be [samples, time steps, features]
  trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
  testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
  # create and fit the LSTM network
  model = Sequential()
  model.add(LSTM(64, return_sequences=True, input_shape=(1, trainX.shape[2])))
  model.add(LSTM(64)) 
  model.add(Dense(1))
  model.compile(loss='mean_squared_error', optimizer='adam')
  es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100)
  history = model.fit(trainX, trainY, epochs=1000, batch_size=10, validation_data=(testX, testY), verbose=1, shuffle=False, callbacks=[es])
  # Predict
  fore_var = pd.read_csv("For_Forecast/all_forecast_price_new.csv")
  pred_var = fore_var[(fore_var['year'] == next_year) & (fore_var['month'] == cvt_mth[mth])]
  pred_var = np.array(pred_var[exogenous_features])
  pred_data = np.reshape(pred_var, (pred_var.shape[0], 1, pred_var.shape[1]))
  predict_price = model.predict(pred_data)
  predict_price = scaler.inverse_transform(predict_price)
  predict_price[predict_price<0] = 0
  return predict_price[0][0]


app = Flask(__name__)
app.config['DEBUG'] = True

@app.route('/', methods = ["GET","POST"])
def index():
  if(request.method == "POST"):
    prd = request.form['product']
    prv = request.form['province']
    area = request.form['area']
    mth = request.form['month']
    if prd == "อ้อยโรงงาน":
      if cvt_mth[mth] in np.arange(4,12):
        return render_template("main_page.html",year = next_year, prediction = "ในช่วงเดือนเมษายน ถึงเดือนพฤศจิกายนจะไม่มีการเก็บเกี่ยวอ้อย")
      else:
        aop = AOP_LR(prd, prv, mth)
        price = PRICE_LSTM(prd, prv, mth)
        price = price*cvt_kg[prd]
        profit = aop*float(area)*price
        return render_template("main_page.html",year = next_year, prediction = f'''เดือน{mth} {prd}ของจังหวัด{prv}จะมีปริมาณผลผลิต {add_comma(round(aop,2))} ตันต่อไร่ 
                และจะมีราคาประมาณ {add_comma(round(price,2))} บาทต่อตัน
                พื้นที่ {area} ไร่ จะขาย{prd}ได้ {add_comma(round(profit,2))} บาท''')
    else:
      aop = AOP_LR(prd, prv, mth)
      price = PRICE_LSTM(prd, prv, mth)
      price = price*cvt_kg[prd]
      profit = aop*float(area)*price
      return render_template("main_page.html",year = next_year, prediction = f'''เดือน{mth} {prd}ของจังหวัด{prv}จะมีปริมาณผลผลิต {add_comma(round(aop,2))} ตันต่อไร่ 
      และจะมีราคาประมาณ {add_comma(round(price,2))} บาทต่อตัน
      พื้นที่ {area} ไร่ จะขาย{prd}ได้ {add_comma(round(profit,2))} บาท''')
  else:
    return render_template("main_page.html", year = next_year, prediction = "")


if __name__ == '__main__':
    app.run(debug = True)