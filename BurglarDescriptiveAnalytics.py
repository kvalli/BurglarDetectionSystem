import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("BurglarTrackingData.csv")

#plot Noise Motion_Presence Luminicence
fig = plt.figure()
ax = fig.add_subplot(2,2,1)
plt.title("Distribution of Noise Motion_Presence Luminicence data")
plt.xlabel("Noise")
plt.ylabel("Values")

ax.plot(df['Noise'])
ax = fig.add_subplot(2,2,2)
ax.plot(df['Motion_Presence'])
plt.xlabel("Motion_Presence")
plt.ylabel("Values")

ax = fig.add_subplot(2,2,3)
plt.xlabel("Luminicence")
plt.ylabel("Values")

ax.plot(df['Luminicence'])
plt.show()

#plot burglar_detected
plot = sns.displot(data = df['burglar_detected'], kind="kde")
plt.title("Distribution of burglar_detected  data")
plt.xlabel("burglar_detected")
plt.ylabel("Values")
plt.show()

#burglar_detected vs Room
sns.barplot(x="burglar_detected" ,y="Room",data=df)
plt.title("Distribution of burglar_detected  vs Room data")
plt.xlabel("burglar_detected")
plt.ylabel("Room")
plt.show()

#burglar_detected vs Noise
sns.barplot(x="burglar_detected" ,y="Noise",data=df)
plt.title("Distribution of burglar_detected  vs Room Noise")
plt.xlabel("burglar_detected")
plt.ylabel("Noise")
plt.show()

#burglar_detected vs Luminicence
sns.barplot(x="burglar_detected" ,y="Luminicence",data=df)
plt.title("Distribution of burglar_detected  vs Room Luminicence")
plt.xlabel("burglar_detected")
plt.ylabel("Luminicence")
plt.show()

#burglar_detected vs Luminicence
sns.boxplot(x="burglar_detected" ,y="Luminicence",data=df)
plt.title("burglar_detected  vs Room Luminicence")
plt.xlabel("burglar_detected")
plt.ylabel("Luminicence")
plt.show()

#burglar_detected vs Noise
sns.boxplot(x="burglar_detected" ,y="Noise",data=df)
plt.title("Distribution of burglar_detected  vs Room Noise")
plt.xlabel("burglar_detected")
plt.ylabel("Room Noise")
plt.show()

#burglar_detected vs Room
sns.boxplot(x="burglar_detected" ,y="Room",data=df)
plt.title("Distribution of burglar_detected  vs Room id")
plt.xlabel("burglar_detected")
plt.ylabel("Room")
plt.show()

#Motion_Presence vs burglar_detected
sns.barplot(x="burglar_detected" ,y="Motion_Presence",data=df)
plt.title("Distribution of burglar_detected  vs Motion_Presence")
plt.xlabel("burglar_detected")
plt.ylabel("Motion_Presence")
plt.show()

sns.boxplot(x="burglar_detected" ,y="Motion_Presence",data=df)
plt.title("Distribution of burglar_detected  vs Motion_Presence")
plt.xlabel("burglar_detected")
plt.ylabel("Motion_Presence")
plt.show()

sns.violinplot(x="burglar_detected" ,y="Motion_Presence",data=df)
plt.title("Distribution of burglar_detected  vs Motion_Presence")
plt.xlabel("burglar_detected")
plt.ylabel("Motion_Presence")
plt.show()

