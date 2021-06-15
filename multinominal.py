############################# problem 1 ########################
### Multinomial Regression ####
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

Mdata = pd.read_csv("C:/Users/usach/Desktop/Multinominal regression/mdata.csv")
Mdata.head(10)

Mdata.columns="sn","id","female","ses","schtyp","prog","read","write","math","science","honors"

Mdata1 = Mdata.drop(["sn","id"],axis=1)

Mdata1.describe()
Mdata1.prog.value_counts()

# Boxplot of independent variable distribution for each category of choice 
sns.boxplot(x="prog",y="read",data=Mdata1)
sns.boxplot(x="prog",y="write",data=Mdata1)
sns.boxplot(x="prog",y="math",data=Mdata1)
sns.boxplot(x="prog",y="science",data=Mdata1)


# Scatter plot for each categorical choice of car
sns.stripplot(x="prog",y="read",jitter=True,data=Mdata1)
sns.stripplot(x="prog",y="write",jitter=True,data=Mdata1)
sns.stripplot(x="prog",y="math",jitter=True,data=Mdata1)
sns.stripplot(x="prog",y="science",jitter=True,data=Mdata1)

# Scatter plot between each possible pair of independent variable and also histogram for each independent variable 
sns.pairplot(Mdata1,hue="prog") # With showing the category of each car choice in the scatter plot
sns.pairplot(Mdata1) # Normal

# Correlation values between each independent features
Mdata1.corr()


train,test = train_test_split(Mdata1,test_size = 0.2)

# ‘multinomial’ option is supported only by the ‘lbfgs’ and ‘newton-cg’ solvers
model = LogisticRegression(multi_class="multinomial",solver="newton-cg").fit(train.iloc[:,4:7],train.iloc[:,3])

test_predict = model.predict(test.iloc[:,4:7]) # Test predictions
help(LogisticRegression)

# Test accuracy 
accuracy_score(test.iloc[:,3],test_predict) # 60%


train_predict = model.predict(train.iloc[:,4:7]) # Train predictions 
# Train accuracy 
accuracy_score(train.iloc[:,3],train_predict) # 59.3%

########################## problem 2 ########################
### Multinomial Regression ####
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

loan = pd.read_csv("C:/Users/usach/Desktop/Multinominal regression/loan.csv")
#2.	Work on each feature of the loanset to create a loan dictionary as displayed in the below image
#######feature of the loanset to create a loan dictionary
loan_details =pd.DataFrame({"column name":loan.columns,
                            "loan type(in Python)": loan.dtypes})

#3.	loan Pre-loancessing
#3.1 loan Cleaning, Feature Engineering, etc
#details of loan 
loan.info()
loan.describe()         
#loan types        
loan.dtypes
#checking for na value
loan.isna().sum()
loan.isnull().sum()
# The column int_rate is character type, let's convert it to float
loan['int_rate'] = loan['int_rate'].apply(lambda x: pd.to_numeric(x.split("%")[0]))

loan.isnull().sum()/len(loan)*100
#droping nall columns form df
df1= loan.dropna(axis=1)
"""	Exploratory Data Analysis (EDA):
	Summary
	Univariate analysis
	Bivariate analysis """
EDA ={"column ": loan.columns,
      "mean": loan.mean(),
      "median":loan.median(),
      "mode":loan.mode(),
      "standard deviation": loan.std(),
      "variance":loan.var(),
      "skewness":loan.skew(),
      "kurtosis":loan.kurt()}

EDA
# covariance for loan set 
covariance = loan.cov()
covariance
# Correlation matrix 
co = loan.corr()
co
#checking unique value for each columns
loan.nunique()
#droping high variance data and single value in columns
df= df1.drop(["id","member_id","earliest_cr_line","pymnt_plan","url","issue_d","addr_state","earliest_cr_line","zip_code","initial_list_status","policy_code","application_type","acc_now_delinq","delinq_amnt"],axis=1)

df.columns
df.head(10)
df.describe()
df.columns
df.loan_status.value_counts()
df.term.value_counts()
df.grade.value_counts()
df.home_ownership.value_counts()
df.verification_status.value_counts()
df.nunique()
# creating  different  df  for discrete and continuous data

df_discrete =df.iloc[:,[3,6,7,8,10,12,14]]

target=df.iloc[:,[11]]

df_continuous=df.iloc[:,[0,1,2,4,5,9,13,15,16,17,18,19,20,21,22,23,24,25,26,27,29]]

"""
#dummy variable creation 
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()

df_discrete['term'] = LE.fit_transform(df_discrete['term']) """
# Create dummy variables on categorcal columns
df_discrete=pd.get_dummies(df_discrete)
def norm_func(i):
    x = (i-i.mean())/(i.std())
    return (x)
df_norm  =  norm_func(df_continuous)
df_norm.describe()
df_continuous.dtypes

model_df = pd.concat([ target,df_norm,df_discrete], axis =1)
model_df.columns
# Scatter plot between each possible pair of independent variable and also histogram for each independent variable 
sns.pairplot(df)
sns.pairplot(df, hue = "loan_status") # With showing the category of each car loanyn in the scatter plot
#boxplot for every columns
df_continuous.columns

loan.boxplot(column=['loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'int_rate',
       'installment', 'annual_inc', 'dti', 'inq_last_6mths', 'open_acc',
       'pub_rec', 'revol_bal', 'total_acc', 'out_prncp', 'out_prncp_inv',
       'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int',
       'total_rec_late_fee', 'recoveries', 'last_pymnt_amnt'])   #no outlier
"""5.	Model Building
5.1	Build the model on the scaled data (try multiple options)
5.2	Perform Multinomial Regression Modl.
5.3	Train and Test the data and compare accuracies by Confusion Matrix, plot ROC AUC curve.
  5.4 Briefly explain the model output in the documentation.
6.	Share the benefits/impact of the solution - how or in what way the business (client) gets benefit from the solution provided.
"""

train, test = train_test_split(model_df, test_size = 0.2 , random_state = 77)

# ‘multinomial’ option is supported only by the ‘lbfgs’ and ‘newton-cg’ solvers
model = LogisticRegression(multi_class = "multinomial", solver="newton-cg").fit(train.iloc[:, 1:],train.iloc[:, 0])

test_predict = model.predict(test.iloc[:, 1:]) # Test predictions
# Test accuracy 
accuracy_score(test.iloc[:,0], test_predict)

train_predict = model.predict(train.iloc[:, 1:]) # Train predictions 
# Train accuracy 
accuracy_score(train.iloc[:,0], train_predict)  

#accuracy is 100%

