import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam

class lstm_model():
  def __init__(self, weights):
    self.model = Sequential()
    # 3 lớp LSTM chồng nhau có trả về sequence
    self.model.add ( LSTM(units=512, return_sequences=True, input_shape=(9,1)))
    self.model.add (Dropout(0.2))
    self.model.add ( LSTM(units=512, return_sequences=True))
    self.model.add (Dropout(0.2))
    self.model.add ( LSTM(units=512, return_sequences=True))
    self.model.add (Dropout(0.2))
    # 1 lớp LSTM không trả về sequence
    self.model.add ( LSTM(units=128, return_sequences=False))
    self.model.add (Dropout(0.2))

    # Đưa qua 1 lớp Dense
    self.model.add ( Dense(units=512))
    # Lớp output ra kết quả
    self.model.add ( Dense(units=11, activation = "softmax"))

    optimizer = Adam()
    self.model.compile(loss = "categorical_crossentropy", metrics=["accuracy"], optimizer=optimizer)

    self.model.load_weights(weights)
  def predict(self, type, input):
    if type == 'price':
      splitedInput = [int(c) for c in input[1:]]
      # print(splitedInput)

      result = self.model.predict(np.expand_dims(splitedInput, axis=0))

      return np.argmax(result)
    if type == 'career':
      career_list = {
        'Viettel': [
          '086',
          '096',
          '097',
          '098',
          '032',
          '033',
          '034',
          '035',
          '036',
          '037',
          '038',
          '039'
        ],
        'Mobifone': [
         '089',
         '090',
         '093',
         '070',
         '079',
         '077',
         '076',
         '078'
        ],
        'Vinaphone': [
          '088',
          '091',
          '094',
          '083',
          '084',
          '085',
          '081',
          '082'
        ],
        'Vietnamobile':[
          '092',
          '056',
          '058'
        ],
        'Wintel':[
          '055'
        ]
      }
      number_head = input[:3]

      for key in career_list:
        if number_head in career_list[key]:
          return key
        
      return 'Unknown Career'