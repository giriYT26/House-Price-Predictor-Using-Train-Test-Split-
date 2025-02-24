#Creating a house price predictor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error

#creating a dataset using the datafram class from pandas
data = {
    "Square_Feet": [1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000],
    "Bedrooms": [2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 8],
    "Bathrooms": [1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7],
    "Price": [200000, 250000, 300000, 350000, 400000, 450000, 500000, 550000, 600000, 650000, 700000]
}
df=pd.DataFrame(data)
print(df)

x=df.drop("Price",axis=1) #independent var (squarefeet,bedroom,bathroom)
y=df["Price"] #dependent var (price)
#splitting the data to do 80% train and 20%test
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

print(f"trained data:{x_train.shape}") #i.e it returns (8,3) #the 8 means the number to be trained here it is from 0-7 and the 3 is the numnber of column
print(f"tested_data:{x_test.shape}") #i.e it retunrs (2,3) #the 2 means the number data to be trained here it is from 7-9 and the 3 is the numeber of column

#creating a LinearRegression module and train the module
model=LinearRegression()
model.fit(x_train,y_train)

#make perdiction 
y_pred=model.predict(x_test)
#display the actual value vs the perdicted value
result=pd.DataFrame({"Actual":y_test,"predicted":y_pred})

#Evaluating the module
MAE=mean_absolute_error(y_test,y_pred)
MSE=mean_squared_error(y_test,y_pred)
RMSE=np.sqrt(MSE)

print(f"The Mean absoulte error is :{MAE}\nThe Mean squared error is : {MSE}\nThe Root mean squared error : {RMSE}")

#Graph to see the resultes
plot.scatter(y_test,y_pred,color="blue",label="Predicted Points")
plot.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle="dashed", label="Ideal Fit (y=x)")
plot.xlabel("Actual Price")
plot.ylabel("Predicted Price")
plot.title("Actual VS Predicted House Prices")
plot.legend()
plot.show()
