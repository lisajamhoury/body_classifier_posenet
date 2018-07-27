# Gesture Classifier Using Posenet, TensorFlow.js and Ml5.js 

:rainbow: Based on Daniel Shiffman's [Color Classifier Example](https://github.com/CodingTrain/website/tree/master/Courses/intelligence_learning/session7). :rainbow: 
:pray: Thanks to Yining for [Posenet + KNN Example](https://github.com/yining1023/posenet-knn).:pray:

Trains model based on recorded posenet data in JSON file. (Will upload the recording code soon). 

User has two options: 

1. Train model with training data in JSON file, then test based on live input using Posenet.  

```
// to test on live input set testLive on line 18 to true
let testLive = true;
```

2. Train model with training data in JSON file, then test the model based on the recorded training data. Bad practice, but we're learning here! 

```
// to test from recorded data set testLive  on line 18 to false
let testLive = false; 
```

### The current poses are: 

1. T pose: Stand with legs together and arms out in a T
2. Boss pose: Stand with legs together with hands on hips
3. X pose: Stand with legs wide apart and arms over head and wide apart so you look like an X

### I included two recordings:

1. fullBodyLisa.json I recorded standing far away from camera. Legs are in full view.
2. closeBodyLisa.json I recorded from about two feet from camera. Legs are not in full view.

### Results

It works! It's not so bad considering my data is not so good (see below). I'm looking forward to testing with better data.

[Add image here]


### Issues 

1. I recorded data in my tiny apartment with bad light and it's not very good. Will record better data at ITP with more space. Maybe today!
2. I'm getting an `identifier has already been used` error at the top of my sketch.js file. It's not affecting anything, but would like to figure out where it's coming from. [Filed issue here](https://github.com/lisajamhoury/body_classifier_posenet/issues/1).

### What I Want To Do Next 

1. Add the recording functionality to repo
2. Train from live data. I believe I just need to call model.fit each time a user adds data to live train it. Is this correct?
3. Record new data. 





