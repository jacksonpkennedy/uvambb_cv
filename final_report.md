Team: Nathan Todd (ygu6ax), Nathan Wan (xsj5rs), Jackson Kennedy (rwv7yy), Hudson Noyes (qmq6gg), James Sweat (jes9hd)

Title: Beyond the Box Score: Deriving Advanced Metrics from NCAAM Broadcast Footage with Computer Vision
Category: Convolutional Neural Networks for Player/Ball Tracking
Motivation: What problem are you tackling? Is this an application or a theoretical result?
	The UVA men’s basketball team–not unlike, presumably, most other NCAA D-I programs–spends several hours weekly manually tracking stats that don’t appear in the box score or play-by-play tables. This may seem archaic, but it’s the only way for coaches to get data on the thing they care about most: their team’s process. Simple plays that often swing possessions but don’t lead immediately to a shot attempt, turnover or rebound aren’t recorded, and thus either go unmeasured and assessed based purely on the “eye test,” or are left up to the assistant coaches and GAs to tediously track by hand. The goal of this project is to automate that process for at least one metric; given UVA’s emphasis on offensive rebounding, we’ll begin by tracking player-level crash rates, i.e., the frequency with which players “crash” the offensive glass after a teammate’s 3-point attempt.
Detection Dataset: Presenting a URL to a dataset you found. Describe the dataset with some basic statistics.
	Our dataset will be a custom-labeled set, beginning with real film from practices/games and then labeling the players, ball, referees and hoop. In addition, there will be a separate set of labeled data for jersey number detection. Our dataset includes 653 labeled images from various snapshots of basketball film with the labeled classes player, ball, referee, and hoop. Our labeled dataset can be found in Google drive here. 
Using finetuned YOLO + SAHI model, auto-label 3-frame temporal windows for Tracknet heatmap probability-based ball localization for better ball capture, as current implementation’s main weakness is ball tracking. Once we have reliable tracking the rest of our methodology and plan of approach remains the same. Progress / data can be found on GitHub.
Video Statistics:
Game
Resolution
FPS
Frames
Duration
Size
game_01
1280x720
60
338,184
94.0 min
1.6 GB
game_02
1280x720
60
440,196
122.3 min
2.2 GB
game_03
1280x720
60
434,816
120.8 min
2.1 GB
Total




1,213,196
337.0 min (~5.6 hrs)
~5.9 GB


Annotated Training Dataset:
488 source images, 5,908 total annotations
4 classes: basketball (90), hoop (490), player (4,314), referee (1,014)
~48:1 player-to-basketball ratio (addressed with 8x oversampling)


Related Work: At least two examples of prior methodology on the topic is a valuable addition.
Hoops Radar: Player Tracking with NBA Broadcast Footage – Tomas Coghlan, Stanford Dept. of Computer Science
Coghlan trains three YOLOv8 models to track players, the ball, and court markings separately, and then, via a homography transformation, maps player and ball positions onto a to-scale, 2-dimensional court representation. Notably, player detections are fed to ByteTrack, which uses Kalman filtering to maintain consistent IDs through occlusions.
Computer Vision-Driven Framework for IoT-Enabled Basketball Score Tracking – Ćirić, I., Ivačko, N., Milić, M., Ristić, P., & Krstić, D.
The authors use a YOLO model to track made baskets, demonstrating that computer vision can automate basic basketball statkeeping tasks. Their model produced an overall accuracy of 88%, just below our 90-95% goal but still a solid benchmark.
Technical Plan: What are the inputs and outputs of your task? Which deep learning models and loss functions do you plan to use?
The input to our task is a video from either a NCAA basketball practice or game. The goal is to implement a series of deep learning models to build out this functionality. First, each image will be put through a model to track the players, referees, ball, and hoop. Then, each player will be put through a model to identify their jersey number. After we have these models in series to track each player, we can then setup a series of rules / criteria to measure our desired metrics, such as a 3-point shot, crashing the board, etc. The outputs will hopefully be a table of statistics tracked over the entire game. Our loss function will be a custom one, though object tracking and memory accuracy play a role, we want it to be based around how accurate the end result of tracking plays and metrics rather than the actual performance of a model, aka we are evaluating the entire pipeline not a specific model.
Evaluation Plan: What experiments are you planning to run? How do you plan to evaluate your machine learning algorithm?
	We will begin by finding clips of film that include various metrics, such as a clip of crashing the board, and feeding those into the model to see if our end model can detect the metrics we are looking to track. On a smaller scale, we want to look at various metrics from each step, such as the probabilities that individual models produce to identify the ball and players, identify their jersey numbers/teams, and also track their movements through time. These smaller metrics, paired with anecdotal verification, will help us evaluate the true performance of our system. Our goal is to reach a 95% accuracy for at least one non-box metric to allow coaches to implement our system with confidence. 
