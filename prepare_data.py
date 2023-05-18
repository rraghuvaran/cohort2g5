import wget
import splitfolders
import zipfile
file_name = 'UNIMIB_2016.zip'
wget.download('https://cdn.extras.talentsprint.com/ADSMI/Datasets/UNIMIB_2016.zip', './')
zip_ref = zipfile.ZipFile(file_name) # create zipfile object
zip_ref.extractall('./') # extract file to dir
zip_ref.close() # close file
splitfolders.ratio('./UNIMIB2016-images', output="data", seed=1337, ratio=(.8, 0.1,0.1)) 
