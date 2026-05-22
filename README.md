# Instrument_Classification_CNN

## Introduction
This personal project uses TensorFlow to build, train and evaluate a classification model which is able to distinguish between audio clips of ten different instruments. Sound classification via Machine Learning is an extremely powerful tool with many real-world use cases such as voice recognition and speech-to-text translation. 

The specific scope of this project revolves around classifying sounds that would be used in a DAW (digital audio workstation) and as a producer myself would find useful in particular for organizing large directions of various sounds (more in the Real-World Application section).

## Background
## FFT and STFT
Audio signals contain a large spectrum of freqeuncies which together form a collective sound. The **Fourier Transform** allows for the decomposion of a signal into its constituent frequencies, creating a spectrum. In particular for signal processing, the **Fast Fourier Transform (FFT)** is commonly used as a fast and effective way to compute the Fourier transform. However this spectrum is still not enough for an ML model to effectively distinguish between different types of sounds as we need to consider how theses frequency varies over time. 

The solution to this is the **Short-Time Fourier Transform (STFT)**, which takes the FFT at stepped fixed-sized windows of the signal over time and then stacks them on top of one another to create a visual representation of the sound over time including variations of amplitude of different frequencies. 


### Mel Spectrogram

## Data Collection and Preprocessing

## Model

## Training and Evaluation

## Real-World Application: Sorting by Instrument Class

## Conclusion

## Sources
https://medium.com/analytics-vidhya/understanding-the-mel-spectrogram-fca2afa2ce53
