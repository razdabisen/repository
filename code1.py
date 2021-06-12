import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import max_error as me

auto=pd.read_csv('Downloads/Automobile_data.csv')
cate=[]
list_num=[]
for i in list(auto.columns):
    if auto[i].dtype=='object':
        cate.append(i)
    else:
        list_num.append(i)
auto['normalized-losses']=auto['normalized-losses'].str.replace('?','145')
auto['normalized-losses']=auto['normalized-losses'].astype('int64')
auto['bore']=auto['bore'].str.replace('?','3.35')
auto['bore']=auto['bore'].astype('float')
auto['stroke']=auto['stroke'].str.replace('?','3.15')
auto['stroke']=auto['stroke'].astype('float')
auto['horsepower']=auto['horsepower'].str.replace('?','130')
auto['horsepower']=auto['horsepower'].astype('int64')
auto['peak-rpm']=auto['peak-rpm'].str.replace('?','4500')
auto['peak-rpm']=auto['peak-rpm'].astype('int64')

for i in list(auto.columns):
    if auto[i].dtype=='object':
            auto[i]=LabelEncoder().fit_transform(auto[i])
output0=auto['prices']
input0=auto.drop(['prices'],axis=1)

input1,input2,output1,output2=train_test_split(input0,output0,random_state=0,test_size=0.2)

lir=LinearRegression()
lir.fit(input1,output1)
lir_pr=lir.predict(input2)
liraccuracy=r2_score(output2,lir_pr)
plt.figure(figsize=(8,5))
sns.scatterplot(x=output2,y=lir_pr)
plt.title('Actual price of test data vs Predicted price of test data') #title
plt.xlabel('Actual price(in $)')
plt.ylabel('Predicted price(in $)')
plt.show()

tr_data_plot = lir.predict(input1)
plt.figure(figsize=(7,7))
ax1 = sns.distplot(output1, hist = True, color = 'r', label = "Actual price of train data")
ax2 = sns.distplot(tr_data_plot, hist = True, color = 'b', label = "Predicted price of train data")
plt.title('Actual price vs Predicted price')
plt.xlabel('Price in $')
plt.ylabel('Proportion of Cars')
plt.legend()
plt.show()
plt.close()
plt.figure(figsize=(7,7))
ax3 = sns.distplot(output2, hist = True, color = 'r', label = "Actual price of test data")
ax4 = sns.distplot(lir_pr, hist = True, color = 'b', label = "Predicted price of test data")
plt.title('Actual price vs Predicted price')
plt.xlabel('Price in $')
plt.ylabel('Proportion of Cars')
plt.legend()
plt.show()
plt.close()

dtr=tree.DecisionTreeRegressor(random_state=0)
dtr=dtr.fit(input1,output1)
dtr_pr=dtr.predict(input2)
dtraccuracy=r2_score(output2,dtr_pr)
sns.scatterplot(x=output2,y=dtr_pr)
plt.title('Actual price of test data vs Predicted price of test data') #title
plt.xlabel('Actual price(in $)')
plt.ylabel('Predicted price(in $)')
plt.show()

tr_data_plot = dtr.predict(input1)
plt.figure(figsize=(7,7))
ax1 = sns.distplot(output1, hist = True, color = 'r', label = "Actual price of train data")
ax2 = sns.distplot(tr_data_plot, hist = True, color = 'b', label = "Predicted price of train data")
plt.title('Actual price vs Predicted price')
plt.xlabel('Price in $')
plt.ylabel('Proportion of Cars')
plt.legend()
plt.show()
plt.close()
plt.figure(figsize=(7,7))
ax3 = sns.distplot(output2, hist = True, color = 'r', label = "Actual price of test data")
ax4 = sns.distplot(lir_pr, hist = True, color = 'b', label = "Predicted price of test data")
plt.title('Actual price vs Predicted price')
plt.xlabel('Price in $')
plt.ylabel('Proportion of Cars')
plt.legend()
plt.show()
plt.close()

rfg=RandomForestRegressor(random_state=0)
rfg.fit(input1,output1)
rfg_pr=rfg.predict(input2)
rfgaccuracy=r2_score(output2,rfg_pr)
sns.scatterplot(x=output2,y=rfg_pr)
plt.title('Actual price of test data vs Predicted price of test data') 
plt.xlabel('Actual price(in $)')
plt.ylabel('Predicted price(in $)')
plt.show()
tr_data_plot = rfg.predict(input1)
plt.figure(figsize=(7,7))
ax1 = sns.distplot(output1, hist = True, color = 'r', label = "Actual price of train data")
ax2 = sns.distplot(tr_data_plot, hist = True, color = 'b', label = "Predicted price of train data")
plt.title('Actual price vs Predicted price')
plt.xlabel('Price in $')
plt.ylabel('Proportion of Cars')
plt.legend()
plt.show()
plt.close()
plt.figure(figsize=(7,7))
ax3 = sns.distplot(output2, hist = True, color = 'r', label = "Actual price of test data")
ax4 = sns.distplot(rfg_pr, hist = True, color = 'b', label = "Predicted price of test data")
plt.title('Actual price vs Predicted price')
plt.xlabel('Price in $')
plt.ylabel('Proportion of Cars')
plt.legend()
plt.show()
plt.close()

ss=StandardScaler()
input11=ss.fit_transform(input1)
input22=ss.fit_transform(input2)
lor=LogisticRegression(max_iter=1000)
lor.fit(input11,output1)
lor_pr=lor.predict(input22)
loraccuracy=r2_score(output2,lor_pr)
sns.scatterplot(x=output2,y=lor_pr)
plt.title('Actual price of test data vs Predicted price of test data') 
plt.xlabel('Actual price(in $)')
plt.ylabel('Predicted price(in $)')
plt.show()
tr_data_plot = lor.predict(input11)
plt.figure(figsize=(7,7))
ax1 = sns.distplot(output1, hist = True, color = 'r', label = "Actual price of train data")
ax2 = sns.distplot(tr_data_plot, hist = True, color = 'b', label = "Predicted price of train data")
plt.title('Actual price vs Predicted price')
plt.xlabel('Price in $')
plt.ylabel('Proportion of Cars')
plt.legend()
plt.show()
plt.close()
plt.figure(figsize=(7,7))
ax3 = sns.distplot(output2, hist = True, color = 'r', label = "Actual price of test data")
ax4 = sns.distplot(lor_pr, hist = True, color = 'b', label = "Predicted price of test data")
plt.title('Actual price vs Predicted price')
plt.xlabel('Price in $')
plt.ylabel('Proportion of Cars')
plt.legend()
plt.show()
plt.close()

mae(output2,lir_pr)
mae(output2,lor_pr)
mae(output2,dtr_pr)
mae(output2,rfg_pr)
me(output2,lir_pr)
me(output2,lor_pr)
me(output2,dtr_pr)
me(output2,rfg_pr)

n={'symboling':[input("symboling:")],'normalized-losses':[input("normalized-losses:")], 'make':[input("make:")], 'fuel-type':[input("fuel-type:")], 'aspiration':[input("aspiration:")], 'num-of-doors':[input("num-of-doors:")], 'body-style':[input("body style:")], 'drive-wheels':[input("drive-wheels:")], 'engine-location':[input("engine-location:")], 'wheel-base':[input("wheel-base:")], 'length':[input("length:")], 'width':[input("width:")], 'height':[input('height:')], 'curb-weight':[input("curb-weight:")], 'engine-type':[input("engine-type:")], 'num-of-cylinders':[input("num-of-cylinders:")], 'engine-size':[input("engine-size:")], 'fuel-system':[input("fuel-system:")], 'bore':[input("bore:")], 'stroke':[input("stroke:")], 'compression-ratio':[input("compression-ratio:")], 'horsepower':[input("horsepower:")], 'peak-rpm':[input("peak-rpm:")],'city-mpg':[input("city-mpg:")], 'highway-mpg':[input("highway-mpg:")]}
new=pd.DataFrame(n)
if rfgaccuracy>liraccuracy and rfgaccuracy>loraccuracy and rfgaccuracy>dtraccuracy:
    predict_new = rfg.predict(new)
elif dtraccuracy>liraccuracy and dtraccuracy>loraccuracy and dtraccuracy>rfgaccuracy:
    predict_new=dtr.predict(new)
elif loraccuracy>rfgaccuracy and loraccuracy>dtraccuracy and loraccuracy>liraccuracy:
    predict_new=lor.predict(new)
elif liraccuracy>dtraccuracy and liraccuracy>loraccuracy and liraccuracy>rfgaccuracy:
    predict_new=lir.predict(new)
new.insert(25, 'prices', predict_new, True)
new.to_csv('Downloads/Automobile_data.csv', mode='a', index=False, header=False)
