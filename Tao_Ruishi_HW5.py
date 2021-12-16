#Ruishi Tao
#ITP499 Fall2021
#HW5

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# 1.Read the dataset into a dataframe. (1)
df=pd.read_csv('A_Z Handwritten Data.csv')
# 2.Explore the dataset and determine what is the target variable. (1)
print(df.head())
# label is the target variable
# 3.Separate the dataframe into feature set and target variable. (2)
X= df.iloc[:,1:]
y= df.iloc[:,0]
# 4.Print the shape of feature set and target variable. (2)
print(X.shape)
print(y.shape)
# 5.Is the target variable values letters or numbers? (1)
# it's a number

# 6.If it is numbers, then how would you map letters to numbers? Hint: Use a data dictionary (2)
word_dict={0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y',25:'Z'}
# 7.Show a histogram (count) of the letters. (1)
plt.figure(1)
s=sb.countplot(x="label",data=df)
s.set_xticklabels(word_dict.values())
plt.show()
# 8.Display one random letter from the dataset along with its label as the figure title. (2)
random_int = np.random.randint(0, len(y))
letter=X.iloc[random_int, :]
label = y[random_int]
letter=np.array(letter)
letter=letter.reshape(28,28)
plt.title('The letter is ' + word_dict[label])
plt.imshow(letter,cmap="gray")
plt.show()

# 9.Partition the data into train and test sets (70/30). Use random_state = 2021. Stratify it. (1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2021, stratify=y)

# 10.Scale the train and test features. (1)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# 11.Create an MLPClassifier. Experiment with various parameters.  (3)
model = MLPClassifier(hidden_layer_sizes=128, activation="relu", random_state=2021)

# 12.Fit the training data to the model. (1)
model.fit(X_train, y_train)

# 13.Plot the loss curve. (1)
plt.plot(model.loss_curve_)

# 14.Display the accuracy of your model. (1)
y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))

# 15.Plot the confusion matrix along with the letters. (2)
cf_matrix = confusion_matrix(y_test, y_pred)
sb.heatmap(cf_matrix, xticklabels=word_dict.values(), yticklabels=word_dict.values(), cmap='Blues')
plt.show()

# 16.Now, display the predicted letter of the first row in the test dataset. Also display the actual letter. Show both actual and predicted letters (as title) on the image of the letter. (4)
plt.imshow(X_test[0].reshape(28, 28),cmap="gray")
plt.title('The letter is ' + word_dict[y_test.iloc[0]] + '\nThe predicted value is ' + word_dict[y_pred[0]])
plt.show()

# 17.Finally, display the actual and predicted letter of a misclassified letter. (4)
plt.imshow(X_test[150].reshape(28, 28),cmap="gray")
plt.title('The letter is ' + word_dict[y_test.iloc[150]] + '\nThe predicted value is ' + word_dict[y_pred[150]])
plt.show()

plt.imshow(X_test[0].reshape(28, 28),cmap="gray")
plt.title('The letter is ' + word_dict[y_test.iloc[0]] + '\nThe predicted value is ' + word_dict[y_pred[0]])
plt.show()