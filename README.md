# 378-Project
This is a repository for collaboration on our ELEC 378 final project

# Repo Link
The link to the repo can be found [here](https://github.com/ThomasPickle7/378-Project).


# Objective
- The goal of this project was to classify music samples into one of 10 genres. To do this, we were given 800 labeled, 30-second audio clips and 200 unlabeled ones, which were to be used as testing data. The genres included: Rock, Jazz, Pop, Hip-Hop, Reggae, Country, Metal, Disco, Classical, and Blues.

# Pre-reqs
- Due to their size, the directories 'train' and 'test' containing the respective data aren't stored in the repo and must be downloaded and put in the repository manually

# How to run the code
- The features of the model are saved in the 'features_test' and 'features_train' folders. Each set of features is stored in a .csv file. The features are extracted from the audio files in the 'test' and 'train' folders
- Since the features of the data are already extracted, the only thing necessary to interface with is the 'training_model.ipynb' file. this file contains the code to train/test the model.

# Data Exploration
- The data is stored in the 'train' and 'test' folders. Each folder contains 10 subfolders, each corresponding to a different genre of music. Each subfolder contains 100 audio files of 30 seconds each. The audio files are in .wav format.
- we began by listening to a few of the samples. Most were high quality and easy to discern the genre. However there were a few that felt as though they could belong to multiple, or even none of those listed in the assignment description. This was useful to know, as it meant that an imperfect model may be expected.
- Next, we read the files into Python in the form of numpy arrays. Checking the shape revealed that the audio files were 661,504, which is approximately 30 seconds of audio at a sample rate of 22,050 Hz. We also checked the sample rate of the audio files to confirm that it was 22,050 Hz. From this we knew that if we were going to extract features from the audio files, we would either need to reduce the size of the data by downsampling or splitting up the data, or use feature aggregation to reduce the size of the data.
- We then plotted some of the audio file's time and frequency domain representations. This was done to get a better understanding of the data and to see if there were any obvious differences between the genres in the time and frequency domains. We found that the time domain representations were very similar between the genres, but the frequency domain representations were different. This was expected, as things like percussive instruments, rhythm and tempo are more easily discerned in the frequency domain.
**MAKE PLOTS FOR ALL OF THIS STUFF**

# Features

## Extracted Features
- These are features that are extracted through charachteristics of a sample's freuqency domain representation. They are represented as numpy arrays, where the ith element corresponds to the ith frequency band. The value of each element is the feature of the sample in that frequency band.
### centroids
- The spectral centroid indicates at which frequency the energy of a spectrum is centered upon. This is like a weighted mean:
- This is useful for determining where the "center of mass" of the spectrum is, which can determine how "bright" or "dark" a sound is.\
![image](visualizations/spectral-bandwidth-centroid-log-power-spectrogram.png)
### rolloff
- The spectral rolloff is the frequency below which some amount of the total energy of the spectrum is contained, in Librosa's case, 85%.
- Since the rolloff indicates where the majority of the energy of the spectrum is useful in analyzing where the "body" of the sound is.
![image](visualizations/spectral-rolloff-mel-spectrogram.png)
### bandwidth
- The spectral bandwidth is the width of the band of frequencies in which the energy of the spectrum is concentrated.
- This is useful for determining how "sharp" or "dull" a sound is.
![image](visualizations/spectral-bandwidth-centroid-log-power-spectrogram.png)
### contrast
- The spectral contrast is the difference in amplitude between peaks and valleys in a given frequency band.
- This indicates the "sharpness" of the sound, as higher contrast between similar frequencies indicates a sharper sound.
![image](visualizations/spectral-contrast.png)
### Flatness
- The spectral flatness is a measure of how "flat" the spectrum is. A flat spectrum has equal energy at all frequencies. For reference, white noise has a flatness of 1, while a pure tone has a flatness of 0.
- This is useful for determining how "noisy" a sound is.
![image](visualizations/spectral-flatness.png)
### chroma stft
- The chroma stft is a 12-element vector that represents the energy of each of the 12 chroma bands.
- This is useful for determining the "color" of the sound, as it represents the energy of each of the 12 chroma bands.
![image](visualizations
### Zero crossing rate
- The zero crossing rate is the rate at which a signal changes sign.
- This is useful for determining the noisiness of a sound, as noisier sounds tend to have a higher zero crossing rate. It is also useful for determining the pitch of a sound, as the zero crossing rate is higher for higher pitched sounds.
![image](visualizations/
### MFCC
- The Mel-frequency cepstral coefficients are a representation of the short-term power spectrum of a sound. They are derived from the Fourier transform of the sound.
- These are useful for determining the timbre of a sound, as they represent the power spectrum of the sound.
![image](visualizations/mfcc-with-mel-spectrogram.png)
### Tonnetz
- The tonnetz is a 6-element vector that represents the tonal centroid features of a sound. The tonal centroid is the weighted mean of the frequencies of the sound.
- This is useful for determining the tonal characteristics of a sound, as it represents the tonal centroid features of the sound.
![image](visualizations/tonnetz-compared-with-chroma-stft.png)
## Musical Features 
- These features are extracted from the time domain, and are represented as numpy arrays, where the ith element corresponds to the ith time frame. The value of each element is the feature of the sample in that time frame.
### Tempo
- Tempo is the speed at which a piece of music is played, and it's measured in BPM (beats per minute).
- This is useful, as different genres of music tend to have different tempos. For example, classical music tends to have a slower tempo, while rock music tends to have a faster tempo.
![image](visualizations
### Harmonic/Percussive
- Harmonic and percussive components are extracted from the audio signal using the Harmonic-Percussive Source Separation (HPSS) algorithm.
- This is useful for determining the harmonic and percussive components of a sound, as they are often used to distinguish between different genres of music.
![image](visualizations/
### Beat Strength
- Beat strength is a measure of the strength of the beat in a piece of music. By beat stregnth, we mean the degree to which a beat is emphasized compared to the rest of the music.
- This is useful for determining the rhythm features of a sound, as different genres of music will use different time signatures more often and also make use of different rhythm patterns.
![image](visualizations/beat-strength-over-time.png)
## Aggregation
- In order to reduce the size of the series data, would take up thousands of columns, we aggregated the data. We did this by taking the mean, standard deviation, and extrema of each feature. This reduced the size of the data from thousands of columns to well under a hundred, which was far more computationally efficient. In doing so, we lost some information, but felt that the tradeoff was worth it.


## Preprocessing
- After extracting the features, we normalized the data. We did this to ensure that the features were on the same scale, as some features had much larger values than others. This was done after aggregating the data, as normalizing the data before aggregating it would have resulted in the loss of information from the std and extrema features. We used the MinMaxScaler from sklearn to normalize the data, as it scales the data to be between 0 and 1, which is useful for neural networks.



## Feature Selection
- After processing and aggregating the data, we were left with a total of 65 scalar features, derived from 14 spectral features. This seemed like a reasonable number of features to work with, but to be sure, we manually went through the features to see if any could be removed. Some features were removed because they were uniform across all samples. For example, the minimum of the zero crossing rate was zero for every sample, since the zero crossing rate is always positive, so it could be removed. This left us with 60 unique features going into the model.




- Since the models and features were developed in parallel, we were able to see the effects of adding more features on each model, both on our training data and on the test submissions. We Found that adding more features generally improved the performance of the models. This was likely because the features were already aggregated, which reduced the chances of overfitting. We also found it useful to split the testing data into a training and validation set, as this allowed us to see how the models were performing on unseen data. This was useful for tuning the hyperparameters of the models.
![image](visualizations/extended_feature_extraction.png)


- It's worth nothing that although the accuracy of the model only goes up a few percentage points when adding more features on certain models, we believe it was worth the added complexity. We knew that the competition woukld likely be decided by a few percentage points, and that the added data would be worth it, especially considering that the models didn't seem to have slower training times with the added features.

# Model Selection
- We used three different models to find the best predictor for our data. These in cluded Multinomial Logistic Regression, a reguar Neural Network, and a Random Forest Classifier. Below is a description of each model.
## Neural Network
### Description
- One of our models is a Deep Neural Network. Deep Neural Networks are a type of machine learning model that are inspired by the structure of the human brain. They are composed of multiple layers of neurons, each of which takes input data and a set of weights. The neurons then apply an activation function to the weighted sum of the inputs to produce an output. The outputs of the neurons are then passed to the next layer, and so on. The weights of the neurons are learned through a process called backpropagation, which adjusts the weights to minimize the error between the predicted output and the true output. Deep Neural Networks are useful for tasks that require a high degree of nonlinearity, such as image and speech recognition.
### Architecture

### Tuning




## Logistic Regression
### Description
### Architecture
### Tuning




## Random Forest
### Description
### Architecture
### Tuning
## Validation
- We validated each model using an 20/80 split between training and test data. We did this to ensure that the models were generalizing well to unseen data. Additionally, we used a technique called cross-validation to tune the hyperparameters and feature necessity of the model. Cross validations involves running the model multiple times, using new subsets of the data for the train/test split each time, in order to ensure that the model didnt simply overfit to the training data.


# Complete Pipeline
The entire pipeline can be seen in the code that we submitted in the CNN code in that model’s section. But, as a summary, the pipeline could be summarized by the following. 
1. Load the data
2. Extract the features
3. Aggregate the features
4. Normalize the features
5. Split the data into training and validation sets
6. Train the model
7. Validate the model
8. Make predictions on the test data


# Conclusions
## Future Improvements
Something we experimented with but didn't implement in our final solution was preprocessing. The audio clips were given to us in 30 second segments, which we felt was too long. Some improvements would could have made include splitting up the audio clips into smaller segments. One idea we had was to potentially split it into 2 second clips, because the average measure in music is around 2 seconds. But if you think about it intuitively, it actually probably takes a person up to 5 or 7 seconds to recognize the genre of a song. So it would be best to play around with sections of 2 to 7 seconds. This process could have improved the quality of our model greatly, as we'd be able to feed it a larger volume of data.
...WRITE MORE HERE

## Takeaways

## Models we will (should) look into
- SVM
- Logistic Regression
- Neural network
- Random Forest


# Potentially useful links
https://www.mage.ai/blog/music-genre-classification
https://www.analyticsvidhya.com/blog/2022/03/music-genre-classification-project-using-machine-learning-techniques/
https://towardsdatascience.com/music-genre-classification-with-python-c714d032f0d8


# Acknowledgements:
- Members:
    - Max Kuhlman
    - Sam Lim
    - Thomas Pickell
    - Alex Zalles
