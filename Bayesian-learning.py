
import numpy as np
import os



directory='20_newsgroups'
folders_list=sorted(os.listdir(os.path.join(directory)))



directory='20_newsgroups'


#Reading the whole data
def getWholeData(folders_list, directory):
    dataDict={} 
    for folder in folders_list:
        dataDict[folder]=[]
        for file in os.listdir(os.path.join(directory,folder)):
            with open(os.path.join(directory,folder,file),encoding='latin-1') as opened_file:
                dataDict[folder].append(opened_file.read())
    return dataDict

dataDict = getWholeData(folders_list, directory)


def clean(dataDict):
   # Creating list of stop words 
    stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
    symbols = ['<','>','?','.','"',')','(','|','-','#','*','+','\'','&','^','`','~','\t','$','%',"'",'!','/','\\','=',',',':']
    stop_words = stop_words + symbols
    # Common words throughout all docs play no part in classification ,so removing them
    # stop_words+=['subject:','from:', 'date:', 'newsgroups:', 'message-id:', 'lines:', 'path:', 'organization:', 
    #             'would', 'writes:', 'references:', 'article', 'sender:', 'nntp-posting-host:', 'people', 
    #             'university', 'think', 'xref:', 'cantaloupe.srv.cs.cmu.edu', 'could', 'distribution:', 'first', 
    #             'anyone','world', 'really', 'since', 'right', 'believe', 'still', 
    #             "max>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'"]
    vocab={}
# Creating a dictionary of words and their frequency
    for i in range(len(dataDict)): 
        for doc in dataDict[folders_list[i]]: 
            for word in doc.split(): 
                if word.lower() not in stop_words and len(word.lower()) >= 5: #filtering out the words
                    if word.lower() not in vocab:
                        vocab[word.lower()]=1
                    else:
                        vocab[word.lower()]+=1
    return vocab

vocab = clean(dataDict)
# Building vocabulary


# Sorting our dictionary based on top frequency list words
def sort_word_frequency(vocab):
    words_frequency=sorted(vocab.items(), key=lambda kv: kv[1], reverse=True)
    return words_frequency

words_frequency = sort_word_frequency(vocab)
#Building the feature list from vocab which contains highest freqency words

# Choosing top 1500 vocab words as features
dictionary_words = {}
feature_list=[]
i=0
for key in words_frequency:
    if i ==1500:
        break
    dictionary_words[key[0]] = i
    feature_list.append(key[0])
    i+=1


def Y_create(dataDict):
    Y=[] # list of newsgroups 
    for i in range(len(dataDict)):
        for doc in dataDict[folders_list[i]]:
            Y.append(folders_list[i])
    Y=np.asarray(Y)
    return Y
Y = Y_create(dataDict)


def X_create(folders_list, dictionary_words):
    X_createe = []

    for folder in folders_list:
        # Insert each file as a new row 
        for file in os.listdir(os.path.join(directory,folder)):
            # Add a new row for every file
            X_createe.append([0]*len(dictionary_words))
            with open(os.path.join(directory,folder,file),encoding='latin-1') as opened_file:
                for word in opened_file.read().split():
                    if word.lower() in dictionary_words:
                        X_createe[len(X_createe)-1][dictionary_words[word.lower()]] += 1
    return X_createe

X_createe = X_create(folders_list, dictionary_words)
X= np.asarray(X_createe)



# Splitting X and Y into training and testing data
def shuffle_split_data(X, y):
    arr_rand = np.random.rand(X.shape[0])
    split = arr_rand < np.percentile(arr_rand, 70)

    X_train = X[split]
    y_train = y[split]
    X_test =  X[~split]
    y_test = y[~split]

    # print len(X_Train), len(y_Train), len(X_Test), len(y_Test)
    return X_train, y_train, X_test, y_test



x_train,y_train,x_test, y_test = shuffle_split_data(X, Y)



def fitting(x_train,y_train):
    res={}
    res["total_data"]=len(y_train)
    class_labels=set(y_train)
    for label_present in class_labels:
        res[label_present]={}
        rows_present=(y_train==label_present)
        x_train_present=x_train[rows_present]
        y_train_present=y_train[rows_present]
        words_count=0
        for i in range(len(feature_list)):
            res[label_present][feature_list[i]]=x_train_present[:,i].sum()
            words_count+=x_train_present[:,i].sum()
        res[label_present]["total_count"]=words_count
    return res



def cal_probability(x,dictionary,class_present):
    result=np.log(dictionary[class_present]["total_count"])-np.log(dictionary["total_data"])
    for i in range(len(feature_list)):
        word_count_present=dictionary[class_present][feature_list[i]]+1
        total_word_count=dictionary[class_present]["total_count"]+len(feature_list)
        current_word_probability=np.log(word_count_present)-np.log(total_word_count)
        for j in range(int(x[i])): # if the frequency of word in test data point is zero then we wont consider it.
            result+=current_word_probability
    return result


def prediction(X_test,dictionary):
    Y_prediction_res=[]
    num = 0
    for x in X_test:
        best_class=-1000
        best_prob=-1000
        F=True
        classes=dictionary.keys()
        for class_present in classes:
            if class_present=="total_data":
                continue
            class_present_probability=cal_probability(x,dictionary,class_present)
            if(F==True or class_present_probability>best_prob):
                best_class=class_present
                best_prob=class_present_probability
            F=False
        Y_prediction_res.append(best_class)
    return Y_prediction_res


dictionary=fitting(x_train,y_train)
Y_prediction_res=prediction(x_test,dictionary)


from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(Y_prediction_res,y_test))




