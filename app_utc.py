import base64
import uvicorn
from fastapi import FastAPI,File
# import io
import pandas as pd
import numpy as np
# from pathlib import Path
#IMAGE PROCESS
# from keras.preprocessing import image
# from keras.applications.vgg16 import preprocess_input, decode_predictions
# from IPython.display import Image
# import matplotlib.image as mpimg
# from scipy.io.wavfile import read, write
import librosa
import librosa.display
# import IPython
# from IPython.display import Audio
#SCALER & TRANSFORMATION
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.preprocessing import MinMaxScaler
# from keras.utils.np_utils import to_categorical
# from sklearn.model_selection import train_test_split
# from keras import regularizers
# from sklearn.preprocessing import LabelEncoder
#ACCURACY CONTROL
# from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score, roc_curve
# from sklearn.model_selection import GridSearchCV, cross_val_score
# from sklearn.metrics import mean_squared_error, r2_score
# #OPTIMIZER
# from keras.optimizers import RMSprop,Adam,Optimizer,Optimizer, SGD
# #MODEL LAYERS
# from keras import models
# from keras import layers
# import tensorflow as tf
# from keras.applications import VGG16,VGG19,inception_v3
# from keras import backend as K
# from keras.utils import plot_model
# from keras.datasets import mnist
import keras
#SKLEARN CLASSIFIER
# from xgboost import XGBClassifier, XGBRegressor
# from lightgbm import LGBMClassifier, LGBMRegressor
# from catboost import CatBoostClassifier, CatBoostRegressor
# from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import GaussianNB
# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
# from sklearn.ensemble import BaggingRegressor
# from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
# from sklearn.neural_network import MLPClassifier, MLPRegressor
# from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
# from sklearn.linear_model import LinearRegression
# from sklearn.cross_decomposition import PLSRegression
# from sklearn.linear_model import Ridge
# from sklearn.linear_model import RidgeCV
# from sklearn.linear_model import Lasso
# from sklearn.linear_model import LassoCV
# from sklearn.linear_model import ElasticNet
# from sklearn.linear_model import ElasticNetCV
import pickle
from tensorflow import keras
#IGNORING WARNINGS
from warnings import filterwarnings
filterwarnings("ignore",category=DeprecationWarning)
filterwarnings("ignore", category=FutureWarning) 
filterwarnings("ignore", category=UserWarning)
from pydantic import BaseModel


class audio(BaseModel):
   audio : str

app=FastAPI()


compile_metrics = ["accuracy"]
compile_loss = "categorical_crossentropy"
compile_optimizer = "adam"
model = keras.models.load_model("urbantrafficmodelfinal.h5",compile=False)
model.compile(optimizer=compile_optimizer,loss=compile_loss,metrics=compile_metrics)

def export_function(path):
    
    data,sample_rate = librosa.load(path,duration=1.0)
    
    output_One = extract_function(data)
    result = np.array(output_One)
    
    noise_output = noise_function(data)
    output_Two = extract_function(noise_output)
    result = np.vstack((result,output_Two))
    
    stretch_output = stretch_function(data)
    stretch_pitch = pitch_function(stretch_output,sample_rate)
    output_Three = extract_function(stretch_pitch)
    result = np.vstack((result,output_Three))
    
    return result

def extract_function(data):
    
    output_result = np.array([])
    mean_zero = np.mean(librosa.feature.zero_crossing_rate(y=data).T,axis=0)
    output_result = np.hstack((output_result,mean_zero))
    
    stft_output = np.abs(librosa.stft(data))
    chroma_output = np.mean(librosa.feature.chroma_stft(S=stft_output,sr=22050).T,axis=0)
    output_result = np.hstack((output_result,chroma_output))
    
    mfcc_output = np.mean(librosa.feature.mfcc(y=data,sr=22050).T,axis=0)
    output_result = np.hstack((output_result,mfcc_output))
    
    root_output = np.mean(librosa.feature.rms(y=data).T,axis=0)
    output_result = np.hstack((output_result,root_output))
    
    mel_output = np.mean(librosa.feature.melspectrogram(y=data,sr=22050).T,axis=0)
    output_result = np.hstack((output_result,mel_output))
    
    return output_result

def noise_function(data):
    noise_value = 0.009 * np.random.uniform() * np.amax(data)
    data = data + noise_value * np.random.normal(size=data.shape[0])
    
    return data

def stretch_function(data,rate=0.8):
    
    return librosa.effects.time_stretch(data,rate=rate)

def pitch_function(data,sampling_rate,pitch_factor=0.5):
    
    return librosa.effects.pitch_shift(data,sr=sampling_rate,n_steps=pitch_factor)

def shift_function(data):
    shift_range = int(np.random.uniform(-5,5) * 1000)
    
    return np.roll(data,shift_range)

def result(arr):
    dic={}
    dic[0]='Air Conditioner'
    dic[1]='Car Horn'
    dic[2]='Children Playing'
    dic[3]='Dog Bark'
    dic[4]='Drilling'
    dic[5]='Engine Idling'
    dic[6]='Gun Shot'
    dic[7]='Jackhammer'
    dic[8]='Siren'
    dic[9]='Street Music'
    res=""
    for i in range(3):
        res+=dic[arr[i]]+" ,"
    return res





@app.get('/')
def index():
    return {'message': 'Welcome to Urban Traffic Control API'}

@app.post("/predict")
async def predict(file: audio):
    file=file.dict()
    wav_file = open("temp.wav", "wb")
    decode_string = base64.b64decode(file['audio'])
    wav_file.write(decode_string)
    x_Train1 = []
    y_Train1 = []   
    wav_features = export_function("temp.wav")  
    for indexing in wav_features:
        x_Train1.append(indexing)
        y_Train1.append('')

    New_Features_Wav1 = pd.DataFrame(x_Train1)
    # New_Features_Wav1["CATEGORY"] = y_Train1
    Part_X1 = New_Features_Wav1.iloc[:,:].values
    # OHE_Function = OneHotEncoder()
    with open('./scaler.pkl', 'rb') as f:
        Scaler_Function = pickle.load(f)
    Part_X1=Scaler_Function.transform(Part_X1)
    Part_X1 = Part_X1.reshape(Part_X1.shape[0], 162, 1)
    prediction_test_Conv2D1 = model.predict(Part_X1)
    prediction_test_Conv2D_Arg1 = np.argmax(prediction_test_Conv2D1,axis=1)
    print(prediction_test_Conv2D_Arg1)
    res=result(prediction_test_Conv2D_Arg1)
    output={'result':res}
    # return np.int64(result)
    return output