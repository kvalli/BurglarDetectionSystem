import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
import uuid
import statistics

#timestamp generation
ts = datetime.now()
ts_list = []
for j in range(0,7):
    for i in range(1440):
        ts +=timedelta(minutes=i)
        ts_list.append(ts)
    ts +=timedelta(days=j)

#Generate Device ID
device_id  = [str(uuid.uuid1()) for x in range(0,5)]

room_id = np.random.uniform(0,5,size = 10080) # 215-538

#luminecence generation
luminicence =  np.random.uniform(215,538,size = 10080) # 215-538

#noise generation
noise_level = np.random.logistic(loc=40,size=10080) #home noise level 40db

#motion detected activity
motion_detection = np.random.logistic(loc=0.5,size = 10080)

burglar_detect_dict = {"Timestamp":ts_list,"Room": room_id , "Luminicence":luminicence,"Noise":noise_level,"Motion_Presence":motion_detection}
burglar_detect_df = pd.DataFrame(burglar_detect_dict)

motion = list(burglar_detect_df['Motion_Presence'])
noise = list(burglar_detect_df['Noise'])

ts = list(burglar_detect_df['Timestamp'])
l = list(burglar_detect_df["Luminicence"])


burglar_detected =[]
mean_noise = statistics.mean(noise) 
lumi = statistics.mean(burglar_detect_df["Luminicence"])

for i in range(0,10080):
    if motion[i] > 0 and noise[i] < mean_noise :
        tsp = datetime.strptime(str(ts[i]),"%Y-%m-%d %H:%M:%S.%f")
        if tsp.hour >= 22 & tsp.hour < 1:  
            burglar_detected.append(1)
        else:
            burglar_detected.append(0)
    else:
        burglar_detected.append(0)

burglar_detect_df['burglar_detected'] = pd.Series(burglar_detected).values 

#Write Generated train data to csv
burglar_detect_df.to_csv('BurglarTrackingData.csv',index=False)

#Test Data Generation

#luminecence generation
luminicence =  np.random.uniform(215,538,size = 5040) # 215-538

#noise generation
noise_level = np.random.logistic(loc=40,size=5040) #home noise level 40db

#motion detected activity
motion_detection = np.random.logistic(loc=0.5,size = 5040)

test_burglar_detect_dict = {"Timestamp":ts_list[0:5040],"Room": room_id[0:5040] , "Luminicence":luminicence,"Noise":noise_level,"Motion_Presence":motion_detection}
test_burglar_detect_df = pd.DataFrame(test_burglar_detect_dict)

test_motion = list(test_burglar_detect_df['Motion_Presence'])
test_noise = list(test_burglar_detect_df['Noise'])

test_ts = list(test_burglar_detect_df['Timestamp'])
test_l = list(test_burglar_detect_df["Luminicence"])


test_burglar_detected =[]
test_mean_noise = statistics.mean(noise) 
test_lumi = statistics.mean(burglar_detect_df["Luminicence"])
for i in range(0,5040):
    if test_motion[i] > 0 and test_noise[i] < test_mean_noise :
        tsp = datetime.strptime(str(test_ts[i]),"%Y-%m-%d %H:%M:%S.%f")
        if tsp.hour >= 22 & tsp.hour < 1:  #& t.minute < 59 or t.hour < 2 & t.minute < 59:
            test_burglar_detected.append(1)
        else:
            test_burglar_detected.append(0)
    else:
        test_burglar_detected.append(0)

test_burglar_detect_df['burglar_detected'] = pd.Series(test_burglar_detected).values

#Write Generated test data to csv
test_burglar_detect_df.to_csv('BurglarTrackingTestData.csv',index=False)

print("Burglar Tracking  System Data Generated Successfully")
