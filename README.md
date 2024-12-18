# EXNO-6-DS-DATA VISUALIZATION USING SEABORN LIBRARY

# Aim:
  To Perform Data Visualization using seaborn python library for the given datas.

# EXPLANATION:
Data visualization is the graphical representation of information and data. By using visual elements like charts, graphs, and maps, data visualization tools provide an accessible way to see and understand trends, outliers, and patterns in data.

# Algorithm:
STEP 1:Include the necessary Library.

STEP 2:Read the given Data.

STEP 3:Apply data visualization techniques to identify the patterns of the data.

STEP 4:Apply the various data visualization tools wherever necessary.

STEP 5:Include Necessary parameters in each functions.

# Coding and Output:

from google.colab import drive
drive.mount('/content/drive')

import seaborn as sns
import matplotlib.pyplot as plt


x = [1,2,3,4,5]
y = [3,6,2,7,1]

sns.lineplot(x=x,y=y)

<img width="422" alt="image" src="https://github.com/user-attachments/assets/bf9d0091-bca0-4657-9d38-33c901435f0e" />


df = sns.load_dataset("tips")
df


<img width="347" alt="image" src="https://github.com/user-attachments/assets/f56aa643-127e-4f96-9583-4c300dfc52f8" />

sns.lineplot(x="total_bill",y="tip",data=df,hue="sex",linestyle="solid",legend="auto")

<img width="443" alt="image" src="https://github.com/user-attachments/assets/50499d78-9c57-4587-9b1f-7e7f2b6e9662" />

x = [1,2,3,4,5]
y1 = [3,5,2,6,1]
y2 = [1,6,4,3,8]
y3 = [5,2,7,1,4]

sns.lineplot(x=x,y=y1)
sns.lineplot(x=x,y=y2)
sns.lineplot(x=x,y=y3)

plt.title('Multi-Line Plot')
plt.xlabel('X Label')
plt.ylabel('Y Label')

<img width="437" alt="image" src="https://github.com/user-attachments/assets/a4588280-242d-4d3e-9812-20bc12204566" />



import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset('tips')

avg_total_bill = tips.groupby('day')['total_bill'].mean()
avg_tip = tips.groupby('day')['tip'].mean()


plt.figure(figsize=(8,6))

p1 = plt.bar(avg_total_bill.index, avg_total_bill, label = 'Total Bill')
p2 = plt.bar(avg_tip.index,avg_tip,bottom=avg_total_bill,label = 'Tip')

plt.xlabel('Day of the Week')
plt.ylabel('Amount')
plt.title('Average Total Bill and Tip by Day')
plt.legend()


<img width="485" alt="image" src="https://github.com/user-attachments/assets/77de03c6-e8c0-46a4-b3ef-799f2c85991e" />



avg_total_bill = tips.groupby('time')['total_bill'].mean()
avg_tip = tips.groupby('time')['tip'].mean()

<img width="482" alt="image" src="https://github.com/user-attachments/assets/9841ec8a-c716-4ea0-ae6f-a7f981dd925b" />


p1 = plt.bar(avg_total_bill.index, avg_total_bill,label = 'Total Bill', width = 0.4)
p2 = plt.bar(avg_tip.index, avg_tip, bottom = avg_total_bill, label = 'Tip', width = 0.4)

plt.xlabel('Time of Day')
plt.ylabel('Amount')
plt.title('Average Total Bill and Tip by Time of Day')
plt.legend()

<img width="422" alt="image" src="https://github.com/user-attachments/assets/54f94e46-62f1-4106-a5ea-af99026b51d8" />



import seaborn as sns
dt = sns.load_dataset('tips')

sns.barplot(x='day',y='total_bill',hue='sex',data=dt,palette='Set1')

plt.xlabel('Day of teh Week')
plt.ylabel('Total Bill')
plt.title('Total Bill by Day and Gender')

<img width="440" alt="image" src="https://github.com/user-attachments/assets/bd500046-3f69-4121-93ca-724ec3bd6d26" />

 import pandas as pd
from google.colab import files
uploaded = files.upload()
tit = pd.read_csv('/content/titanic_dataset.csv')
tit

<img width="875" alt="image" src="https://github.com/user-attachments/assets/e5eeabbb-92e8-4072-a3f3-1a3eb8d0ad37" />


plt.figure(figsize=(8,5))
sns.barplot(x='Embarked', y='Fare', data = tit, palette = 'rainbow')
plt.title("Fare of Passenger by Embarked Town")

<img width="480" alt="image" src="https://github.com/user-attachments/assets/bc04443c-c529-459c-aca0-72f18327f142" />

plt.figure(figsize=(8, 5)) # Changed figsize(8,5) to figsize=(8,5)
sns.barplot(x='Embarked', y='Fare', data=tit, palette='rainbow', hue='Pclass')
plt.title("FARE OF PASSENGER BY EMBARKED TOWN, DIVIDED BY CLASS")

<img width="485" alt="image" src="https://github.com/user-attachments/assets/36fd3fd4-8cfe-45c6-8181-553035fae16b" />

import seaborn as sns

tips = sns.load_dataset('tips')

sns.scatterplot(x='total_bill',y='tip',hue='sex',data=tips)

plt.xlabel('Total Bill')
plt.ylabel('Tip Amount')
plt.title('Scatter Plot of Total Bill vs. Tip Amount')

<img width="445" alt="image" src="https://github.com/user-attachments/assets/7620ceec-3515-4a54-8313-4cb528e354ce" />

import seaborn as sns
import numpy as np
import pandas as pd

np.random.seed(1)
num_var = np.random.randn(1000)
num_var = pd.Series(num_var, name = "Numerical Variable")
num_var

sns.histplot(data=num_var,kde=True)

<img width="441" alt="image" src="https://github.com/user-attachments/assets/2b9e08ae-6867-4246-9cf4-0e79c6d1ce58" />


<img width="185" alt="image" src="https://github.com/user-attachments/assets/7ee7aa7f-75d2-4755-9737-ef590280336e" />

from google.colab import files
uploaded = files.upload()
df = pd.read_csv('/content/titanic_dataset.csv')
df

<img width="884" alt="image" src="https://github.com/user-attachments/assets/b7a07105-5c9d-4052-9fe6-dc82c07b2ae5" />

sns.histplot(data=df,x="Pclass",hue="Survived",kde=True)

<img width="452" alt="image" src="https://github.com/user-attachments/assets/ccacb98d-0346-4a76-bbac-80089d1285c0" />

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(0)
marks = np.random.normal(loc=70,scale=10,size=100)
marks

<img width="449" alt="image" src="https://github.com/user-attachments/assets/eb626322-8efb-42ed-a7f8-dffe6fd35004" />


sns.histplot(data=marks,bins=10,kde=True,stat='count',cumulative=False,multiple = 'stack',element = 'bars',palette='Set1',shrink=0.7)
plt.xlabel('Marks')
plt.ylabel('Density')
plt.title('Histogram of Students Marks')


<img width="452" alt="image" src="https://github.com/user-attachments/assets/586fdc2e-0c45-49d1-92b3-7797be12cc3e" />

import seaborn as sns
import pandas as pd
tips = sns.load_dataset('tips')
sns.boxplot(x=tips['day'],y=tips['total_bill'],hue = tips['sex'])


<img width="452" alt="image" src="https://github.com/user-attachments/assets/5feed52f-1ce5-4fb5-bcff-1c22ebad0512" />

sns.boxplot(x="day",y="total_bill",hue="smoker",data=tips,linewidth=2,width=0.6,boxprops={"facecolor": "lightblue", "edgecolor":"darkblue"}, whiskerprops={"color":"black", "linestyle":"--","linewidth":1.5},capprops={"color":"black","linestyle":"--","linewidth":1.5})

<img width="440" alt="image" src="https://github.com/user-attachments/assets/b0c17a8f-1b3e-45ef-9144-19e7b47b5753" />

sns.boxplot(x='Pclass',y='Age',data = df, palette = 'rainbow')
plt.title("Age by Passsenger Class, Titanic")

<img width="437" alt="image" src="https://github.com/user-attachments/assets/7f0c6387-693d-47a9-b6ba-d1d56cd484a1" />

sns.violinplot(x="day",y="total_bill",hue="smoker",data=tips,linewidth=2,width=0.6,palette="Set3",inner="quartile")

plt.xlabel("Dat of the Week")
plt.ylabel("Total Bill")
plt.title("Violin Plot of Total Bill by Day and Smoker Status")

<img width="448" alt="image" src="https://github.com/user-attachments/assets/50ddb8e7-a37b-47e9-acb4-d4d1b8fa858b" />


import seaborn as sns
sns.set(style='whitegrid')
tip = sns.load_dataset('tips')
sns.violinplot(x='day',y='tip',data=tip)



<img width="454" alt="image" src="https://github.com/user-attachments/assets/8f68f193-2792-4bb7-aca1-755f32faf319" />

import seaborn as sns
sns.set(style='whitegrid')
tip = sns.load_dataset('tips')
sns.violinplot(x=tip["total_bill"])

<img width="423" alt="image" src="https://github.com/user-attachments/assets/8e558267-bb82-4e74-b8d6-1fdc0b91cd3d" />


import seaborn as sns
sns.set(style="whitegrid")
tips = sns.load_dataset("tips")
sns.violinplot(x="tip",y="day",data=tip)

<img width="455" alt="image" src="https://github.com/user-attachments/assets/a9162fb2-9b95-45cd-a607-9c834b17ebf9" />

sns.kdeplot(data=tips,x='total_bill',hue='time',multiple='fill',linewidth=3,palette='Set2',alpha=0.8)

<img width="457" alt="image" src="https://github.com/user-attachments/assets/774fce47-79e3-4c3c-9049-f5c597c8e3fe" />

sns.kdeplot(data=tips,x='total_bill',hue='time',multiple='layer',linewidth=3,palette='Set2',alpha=0.8)

<img width="463" alt="image" src="https://github.com/user-attachments/assets/ed81f798-3a9b-41c3-b21b-0825612e794d" />

sns.kdeplot(data=tips,x='total_bill',hue='time',multiple='stack',linewidth=3,palette='Set2',alpha=0.8)

<img width="463" alt="image" src="https://github.com/user-attachments/assets/9d1cbfc7-8d97-454f-ac76-58b5716bf364" />

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from google.colab import files
uploaded = files.upload()
mart = pd.read_csv('/content/supermarket.csv')
mart

<img width="884" alt="image" src="https://github.com/user-attachments/assets/26c0e793-8646-4817-8757-c6ae933b05dc" />


mart = mart[['Gender','Payment','Unit price','Quantity','Total','gross income']]
mart.head(10)

<img width="415" alt="image" src="https://github.com/user-attachments/assets/7f78d035-d199-4873-8d2d-150ff588482d" />

sns.kdeplot(data=mart, x='Total')

<img width="476" alt="image" src="https://github.com/user-attachments/assets/69998ab7-7908-4146-9255-d24ccc088c8e" />

sns.kdeplot(data=mart, x='Unit price')

<img width="464" alt="image" src="https://github.com/user-attachments/assets/91459cd7-442b-4f0e-a940-5db58f45de07" />

sns.kdeplot(data=mart)
<img width="465" alt="image" src="https://github.com/user-attachments/assets/d723ed5e-fd2f-49ab-af48-3c98692456eb" />

sns.kdeplot(data=mart,x='Total',hue='Payment',multiple='stack')

<img width="470" alt="image" src="https://github.com/user-attachments/assets/72ae922b-3592-4e02-ba47-623be873f7e3" />

sns.kdeplot(data=mart,x='Unit price',y='gross income')

<img width="467" alt="image" src="https://github.com/user-attachments/assets/1a27949a-0a65-476b-b99c-4496f75abc2a" />

data = np.random.randint(low=1,high=100,size=(10,10))
print("The data to be plotted:\n")
print(data)

<img width="207" alt="image" src="https://github.com/user-attachments/assets/dbb7a209-baf7-4034-9b5f-856362307547" />

hm = sns.heatmap(data=data,annot=True)
<img width="405" alt="image" src="https://github.com/user-attachments/assets/2091c2bf-eed8-4e44-a5fa-bb8ca6409eb5" />

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")

numeric_cols = tips.select_dtypes(include=np.number).columns
corr = tips[numeric_cols].corr()

sns.heatmap(corr,annot=True, cmap="plasma",linewidths=0.5)


<img width="419" alt="image" src="https://github.com/user-attachments/assets/9bbf4ca8-4a53-4470-93c9-3d1371ef7dd9" />

# Result:
 Include your result here
