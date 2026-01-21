***Sign Language Interpretation Project***

This project was prepared using python 3.10, and  this version was found to be most stable with the application.

This project uses CNN architecture along with mediapipe features to properly classify 28 letters of alphabet, and 2 special characters one for space and one for backspace. The signer can perform realtime hand gesture and it will be detected through the application. It focuses on realtime sentence formation for proper and seamless communication. 

The data folder contains sample of each letter and special character, signer can replicate those actions for checking the functionality of the model. 

The signer can also use the dataCollection.py file for collecting his/her own data. You can simply run the dataCollection.py file and press 's' key for capturing images. 

Once your data is collected you can run train.py to train your own model and dump it inside model2 folder.

For running the inference and detecting the sign in real-time user can run test2.py or test3.py file. If the user has not trained his/her own model then they can also download a sample model from this google drive link https://drive.google.com/file/d/1PHBPycOFuyHcJ3hD2jaD1bI9W4Bmi7rF/view?usp=sharing

Simply download this model and put it in your model2 folder, then run the test2.py file. It will give you PyQt5 application window running the inference. Here you can detect your sign and form words and sentences in real-time. 

The words and sentences formed are automatically dumped to sentences.csv file along with the time stamp. User can fetch this information whenever it is needed. 
