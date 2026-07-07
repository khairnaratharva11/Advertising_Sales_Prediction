import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

def ASKAdvertise(DataPath):

    Border = '-'*100
    #----------------------------------------------
    # Step 1 : Load Dataset
    #----------------------------------------------
    print(Border)
    print("Step 1 : Load Dataset")
    print(Border)

    df = pd.read_csv(DataPath)

    print("Few Records from the datasets : ")
    print(df.head())

    #----------------------------------------------
    # Step 2 : Remove Unwanted Columns
    #----------------------------------------------
    print(Border)
    print("Step 2 : Remove Unwanted Columns")
    print(Border)

    print("Shape of dataset before removal : ", df.shape)
    
    if 'Unnamed: 0' in df.columns:
        df.drop(columns= ['Unnamed: 0'], inplace= True)

    print("Shape of dataset after removal : ", df.shape)

    print(Border)
    print("Clean Dataset is : ")
    print(Border)

    print(df.head())

    #----------------------------------------------
    # Step 3 : Check Missing Values
    #----------------------------------------------
    print(Border)
    print("Step 3 : Check Missing Values")
    print(Border)

    print("Missing Values count : \n",df.isnull().sum())

    #----------------------------------------------
    # Step 4 : Display Statistical Summary
    #----------------------------------------------
    print(Border)
    print("Step 4 : Display Statistical Summary")
    print(Border)

    print(df.describe())

    #----------------------------------------------
    # Step 5 : Correlation Between Columns
    #----------------------------------------------
    print(Border)
    print("Step 5 : Correlation Between Columns")
    print(Border)

    print("Correlation Matrix : ")
    print(df.corr())


    #----------------------------------------------
    # Step 6 : Split Dataset into Dependent and Independent Variables
    #----------------------------------------------
    print(Border)
    print("Step 6 : Split Dataset into Dependent and Independent Variables")
    print(Border)

    X = df[['TV','radio','newspaper']]
    Y = df['sales']

    print("Shape of Independent Variables : ", X.shape)
    print("Shape of Dependent Variables : ", Y.shape)


    #----------------------------------------------
    # Step 7 : Split Dataset for Training & Testing
    #----------------------------------------------
    print(Border)
    print("Step 7 : Split Dataset for Training & Testing")
    print(Border)

    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2, random_state=42)

    print("X_train Shape : ", X_train.shape)
    print("X_test Shape : ", X_test.shape)
    print("Y_train Shape : ", Y_train.shape)
    print("Y_test Shape : ", Y_test.shape)

    #----------------------------------------------
    # Step 8 : Create & Train the Model
    #----------------------------------------------
    print(Border)
    print("Step 8 : Create & Train the Model")
    print(Border)

    model = LinearRegression()
    model.fit(X_train,Y_train)


    #----------------------------------------------
    # Step 9 : Test the Model
    #----------------------------------------------
    print(Border)
    print("Step 9 : Test the Model")
    print(Border)

    Y_pred = model.predict(X_test)


    #----------------------------------------------
    # Step 10 : Evaluate the Model
    #----------------------------------------------
    print(Border)
    print("Step 10 : Evaluate the Model")
    print(Border)
    
    MSE = mean_squared_error(Y_test,Y_pred)
    RMSE = np.sqrt(MSE)
    R2 = r2_score(Y_test, Y_pred)

    print("Mean Squared Error : ", MSE)
    print("Root Mean Squared Error : ", RMSE)
    print("R Square Value : ", R2)

    #----------------------------------------------
    # Step 11 : Calculate Model Coefficient
    #----------------------------------------------
    print(Border)
    print("Step 11 : Calculate Model Coefficient")
    print(Border)
    
    for column, value in zip(X.columns, model.coef_):
        print(f"{column} : {value}")
    
    print("Intercept : ", model.intercept_)


    #----------------------------------------------
    # Step 12 : Compare the Actual & Predicted Values
    #----------------------------------------------
    print(Border)
    print("Step 12 : Compare the Actual & Predicted Values")
    print(Border)

    Result = pd.DataFrame({'Actual sale' : Y_test.values,
                          'Predicted Sale ' : Y_pred
                          })

    print(Result.head())

    #----------------------------------------------
    # Step 13 : Plot Actual VS Predicted
    #----------------------------------------------
    print(Border)
    print("Step 13 : Plot Actual VS Predicted")
    print(Border)

    plt.figure(figsize=(8,5))
    plt.scatter(Y_test,Y_pred)
    plt.xlabel("Actual Sales")
    plt.ylabel("Predicted Sales")
    plt.title("Actual Sales VS Predicted Sales")
    plt.grid(True)
    plt.show()
    
def main():
    ASKAdvertise('Advertising.csv')

        
if __name__ == "__main__":
    main()