import requests
import time

'''
your localhost url. If running on port 5000
'''
url = "http://localhost:5000/predict"
# Path to image file
filess = {"img": open("14.jpg", "rb")}
starttime = time.time()
headers = {'Content-type': 'application/json'}
results = requests.post(url, files=filess, headers=headers)
print("time taken:", time.time() - starttime)
print(results.text)


