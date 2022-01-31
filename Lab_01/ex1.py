# Store the measurements in a csv file, where the first column 
# reports the date (in dd/mm/yyyy format), the second the hour 
# (in the hh:mm:ss format), the third the temperature value, 
# and the fourth the humidity value.

# Note : the temperature and the humidity have been recorder through
# a DHT11 Sensor on a Raspberry Pi

import time
import board
import adafruit_dht
import psutil

import datetime 

import pandas as pd

sensor = adafruit_dht.DHT11(board.D4)

frequency = 5
period = 20

dates = []
hours = []
hums = []
temps = []

count = 0

while count <= int(period/frequency):
	try:
		temp = sensor.temperature
		humidity = sensor.humidity

		date = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")

		dates.append(date.split(" ")[0])
		hours.append(date.split(" ")[1])

		temps.append(temp)
		hums.append(humidity)

		count += 1

	except RuntimeError as error:
        	print(error.args[0])
        	continue

	time.sleep(frequency)


df = pd.DataFrame({'dates':dates,'hours':hours, 'temps':temps, 'hums':hums})
print(df)
df.to_csv('dataset_ex1.csv', index=False, header=False)

