// Body Classifier based on 

// Daniel Shiffman
// Intelligence and Learning
// The Coding Train

// Full tutorial playlist:
// https://www.youtube.com/playlist?list=PLRqwX-V7Uu6bmMRCIoTi72aNWHo7epX4L

// Code from end of 7.12
// https://youtu.be/lz2L-sT8bG0

// Community version:
// https://codingtrain.github.io/ColorClassifer-TensorFlow.js
// https://github.com/CodingTrain/ColorClassifer-TensorFlow.js

// decide if you want to test live or test the recorded data
let testLive = true;
let testRecorded = false; 

// 
let data;
let model;
let xs, ys;
let labelP;
let lossP;
let statusMsg;

// video width and height
// matches from recording sketch 
const videoWidth = 600;
const videoHeight = 500;

// place to store skeleton points and pose labels 
let points = [];
let labels = [];

// for testing recorded data after training 
let counter = 0;
let trained = false;

// live posenet variables
let video;
let poseNet;
let poses = [];

function preload() {
  data = loadJSON('fullBodyLisa.json');
}

function setup() {
  
  
  // Crude interface
  labelP = createDiv('Pose guess: still training');
  lossP = createP('Loss: ');

  // get all the points from recorded JSON 
  for (let i =0; i < Object.keys(data).length; i++) {
    let temppoints = data[i].keypoints;
    let pointset = [];

    for (let j = 0; j < temppoints.length; j++ ) {

      // normalize xs and ys with video width and height from precording sketch 
      // all numbers now 0-1
      pointset.push(temppoints[j].position.x / videoWidth);
      pointset.push(temppoints[j].position.y / videoHeight);
    }

    points.push(pointset);
    labels.push(data[i].posenumber);

  }

  // create tensors and model 
  xs = ml5.tf.tensor2d(points);
  let labelsTensor = ml5.tf.tensor1d(labels, 'int32');

  ys = ml5.tf.oneHot(labelsTensor, 3).cast('float32');
  labelsTensor.dispose();

  model = ml5.tf.sequential();
  const hidden = ml5.tf.layers.dense({
    units: 16,
    inputShape: [34],
    activation: 'sigmoid'
  });
  const output = ml5.tf.layers.dense({
    units: 3,
    activation: 'softmax'
  });
  model.add(hidden);
  model.add(output);

  const LEARNING_RATE = 0.25;
  const optimizer = ml5.tf.train.sgd(LEARNING_RATE);

  model.compile({
    optimizer: optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  // train model based on recorded data
  train();

  createCanvas(videoWidth, videoHeight);

  select('#status').html('Now loading Posenet... ');
  // create video for posenet 
  video = createCapture(VIDEO);
  video.size(width, height);

  // create a new poseNet method with a single detection
  poseNet = ml5.poseNet(video, modelReady);

  // this sets up an event that fills the global variable "poses"
  // with an array every time new poses are detected
  poseNet.on('pose', function(results) {
    poses = results;
  });
  
  // Hide the video element, and just show the canvas
  video.hide();

}

// for loading posenet model
function modelReady() {
  select('#status').html('');
}

// this is what trains our model 
// using 1000 epochs works well 
async function train() {
  const epochs = 50;
  // This is leaking https://github.com/tensorflow/tfjs/issues/457
  await model.fit(xs, ys, {
    shuffle: true,
    validationSplit: 0.1,
    epochs: epochs,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        console.log(epoch);
        labelP.html(`Training: epoch ${epoch} of ${epochs}`);
        lossP.html('Loss: ' + logs.loss.toFixed(5));
      },
      onBatchEnd: async (batch, logs) => {
        await ml5.tf.nextFrame();
      },
      onTrainEnd: () => {
        trained = true;
        console.log('finished')
      },
    },
  });
}

function draw() {

  // tests the model based on the data used to train the model
  if (trained && testRecorded) {

    let testInput = points[counter];

    // we can just guess points since the recorded data was processed in setup
    guessPoints(testInput);

  }

  // tests the model based on live data from webcam
  if (trained && testLive) {

    image(video, 0, 0, width, height);

    // this will extract data > draw it > make a prediction
    processPoseData();

  }

}

function processPoseData() {

  for (let i = 0; i < poses.length; i++) {
    let skeleton = poses[i].skeleton;

    drawSkeleton(skeleton);
    // For each pose detected, loop through all the keypoints
    let pose = poses[i].pose;
    let pointsToGuess = [];

    for (let j = 0; j < pose.keypoints.length; j++) {
      // A keypoint is an object describing a body part (like rightArm or leftShoulder)
      let keypoint = pose.keypoints[j];
      drawKeypoint(keypoint);

      pointsToGuess.push(keypoint.position.x / videoWidth, keypoint.position.y / videoHeight);

    }
    guessPoints(pointsToGuess);

  }
}


function drawKeypoint(keypoint) {

  // if confidence is good, draw the joint
  if (keypoint.score > 0.2) {
    fill(255, 0, 0);
    noStroke();
    ellipse(keypoint.position.x, keypoint.position.y, 10, 10);
  }

}



function drawSkeleton(skeleton) {

  // For every skeleton, loop through all body connections and draw them
  for (let j = 0; j < skeleton.length; j++) {
    let partA = skeleton[j][0];
    let partB = skeleton[j][1];
    stroke(255, 0, 0);
    line(partA.position.x, partA.position.y, partB.position.x, partB.position.y);
  }
  
}


function guessPoints(pointsToGuess) {
  
  ml5.tf.tidy(() => {
    const input = ml5.tf.tensor2d([
      pointsToGuess
    ]);
    let results = model.predict(input);
    let argMax = results.argMax(1);
    let index = argMax.dataSync()[0];

    // if running from webcam say guess
    if (testLive) labelP.html(`Pose guess: ${index}`);
    
    // if testing from recorded data, compare the actual pose to the guess
    if (testRecorded) { 
      let currentLabel = '';
      if (counter > 0) currentLabel = labelP.elt.innerHTML;
      
      labelP.html(`${currentLabel} 
        <br>
        Pose guess: ${index} / actual: ${data[counter].posenumber}`);
      // console.log('guess / actual');
      // console.log(index + ' / ' );

      if (counter < points.length-1) counter++; // iterate through all recorded data
      else noLoop(); // stops program at end of test
    }
  
  });

}






