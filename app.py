from flask import Flask, render_template, request, url_for
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import math

#loading the model 
model = tf.keras.models.load_model('ipl_score_prediction.h5')
app = Flask(__name__, template_folder="templates")

# Render homepage
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

# Render Results page
@app.route('/result', methods=['GET', 'POST'])
def result():
    # if request.method == 'POST':
      venue = request.form['match_venue']
      batting_team = request.form['batting_team']
      bowling_team = request.form['bowling_team']
      striker = request.form['striker']
      bowlers = request.form.getlist('bowlers')
      
      array1 = [venue, 1, batting_team, bowling_team, striker]
      input_array = array1 + [bowlers[0]]
      # print('type(input_array)',type(input_array))
      a_file = open("features.pkl", "rb")
      # pickle.dump()
      output = pickle.load(a_file)
      # print()
      features= dict(output)
      a_file.close()
      # print('features ',features)
      
      # print()
      labeled_data=[]
      for i in input_array:
        # print(i, ' ', type(i))
        if type(i)==str or type(i)=='numpy.str_':
            # print('yes it ', i)
            if i in features.keys(): 
                # print(' = ',features[i])
                labeled_data.append(features[i])
        else:
            labeled_data.append(i)
      
      input_array=labeled_data
      scaler = MinMaxScaler()
      # input_array=[15, 1, 7, 13, 259, 30]
      # print()
      input_array = np.array(input_array)
      # print(' input array after ' ,labeled_data,'')
      # print()
      input_array= input_array[:6]
     
      arr = [[1]*6]*1
      arr[0][0] = input_array[0]
      arr[0][1] = input_array[1]
      arr[0][2] = input_array[2]
      arr[0][3] = input_array[3]
      arr[0][4] = input_array[4]

      arr = np.array(arr)
      print("input_array jj ??",arr)
      predicted_score = model.predict(arr)

      #input_array= scaler.fit_transform([input_array])
      #predicted_score = model.predict(input_array)
      predicted_score = np.round_(predicted_score[0], decimals=0)
    
      # Projected scores
    #   predicted_score = math.ceil(predicted_score )  
      crr = (predicted_score/6)
      ps_crr = (predicted_score + 14* crr) # @current run rate

      ps_10rpo = (predicted_score + 140) # @10rpo

      ps_12rpo = (predicted_score + (14*12)) # @12rpo

      return render_template('prediction.html', score = predicted_score, curr_rpo = ps_crr[0], rpo10 = ps_10rpo[0], rpo12 = ps_12rpo[0])

if __name__ == "__main__":
    app.run(debug=True)