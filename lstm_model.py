import numpy as np
import re
import json

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
def is_empty(dict):
  return not bool(dict)
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
    if type == 'features':
      features = {}
      sub_strings = []
      hasRepetition = False
      for i in range(2, 5):
        for j in range(2, len(input)):
          sub_string = input[j: j+i]
          if(len(sub_string) != i or sub_string in sub_strings or len(set(sub_string)) == 1):
            continue
          sub_strings.append(sub_string)
      
      max_appr_pos = []
      max_string_length = -1
      max_total_appr_time = -1
      ft_sub_string = ""
      for sub_string in sub_strings:
        appr_pos = [m.start() for m in re.finditer(sub_string, input[2:])]
        total_appr_time = len(appr_pos)
        str_len = len(sub_string)
        if(str_len > max_string_length and total_appr_time >= 2):
          max_string_length = str_len
          max_total_appr_time = total_appr_time
          max_appr_pos = appr_pos
          ft_sub_string = sub_string
      
      if(max_string_length >= 3):
        hasRepetition = True
      else:
        if(max_total_appr_time >= 3):
          hasRepetition = True
        else:
          for index in range(0, len(max_appr_pos)-1):
            if(max_appr_pos[index+1] - max_appr_pos[index] == 2):
              hasRepetition = True
      if(hasRepetition):
        features["repetition"] = ft_sub_string
              


          
      


      god_and_fengshui = {
        "godOfWealth": ["39", "79"],
        "godOfSoil": ["38", "78"],
        "fortune": ["68", "6688", "666888", "86", "8866", "888666"]
      }
      
      for type in god_and_fengshui:
        for tail in god_and_fengshui[type]:
          if(input[3:].endswith(tail)):
            features[type] = tail

      

      array = [c for c in input[1:]]
      array_dict = {}


      for index in range(0, len(array)):

        rpt_times = 1
        for sub_index in range(index+1, len(array)):
          if(array[index] == array[sub_index]):
            rpt_times += 1
          else:
            break
        if(rpt_times >= 2):
          if(array[index] in array_dict):
            array_dict[array[index]] = max(array_dict[array[index]], rpt_times)
          else:
            array_dict[array[index]] = rpt_times


      if(not is_empty(array_dict)):

        curr_digit = "" #Chữ số có số lần xuất hiện nhiều nhất
        max_time_appr = -1 #Số lần xuất hiện của chữ số đó (đảm bảo có số lần xuất hiện nhiều nhất)
        time_appr_max = -1 #Số chữ số cũng có số lần xuất hiện như trên

        for keys in array_dict:
          if(array_dict[keys] >= max_time_appr):
            max_time_appr = array_dict[keys]
            curr_digit = keys
            time_appr_max = 1
          
          if(array_dict[keys] > max_time_appr):
            time_appr_max = 1
          elif (array_dict[keys] == max_time_appr):
            time_appr_max += 1
        features["nOfAKind"]={
              "kind": [curr_digit],
              "n": max_time_appr,
              "isAtEnd": input.rfind(curr_digit) == len(input)-1
        }
        if(max_time_appr <= 3 and time_appr_max >= 2):
          for keys in array_dict:
            if array_dict[keys] == max_time_appr and keys != curr_digit:
              features["nOfAKind"]["kind"].append(keys)
          
      straight = "" 
      delta = int(array[len(array)-1]) - int(array[len(array)-2])
      if(abs(delta) == 1): 
        straight = array[len(array)-1] + array[len(array)-2]
        for i in range(len(array)-2, 1, -1):
          if int(array[i]) - int(array[i-1]) == delta:
            straight += array[i-1]
      if len(straight) >= 3:
        features["straight"] = straight[::-1]
      return features