# Rock Paper Scissors Computer Vision

> Rock Paper Scissors Model created and trained using teachablemachine. Exported and ran locally to play Rock Paper Scissors using the webcam

## Milestone 2 - Training the model

Created an impage project model with four different classes: Rock, Paper, Scissors and Nothing using Teachable-Machine. This website allowed me to create and train a model using images from my webcam as the training data. The model predicsts the class based on the input of the webcam. 

```python
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Load the model
model = load_model('keras_model.h5')

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
# Replace this with the path to your image
image = Image.open('<IMAGE_PATH>')
#resize the image to a 224x224 with the same strategy as in TM2:
#resizing the image to be at least 224x224 and then cropping from the center
size = (224, 224)
image = ImageOps.fit(image, size, Image.ANTIALIAS)

#turn the image into a numpy array
image_array = np.asarray(image)
# Normalize the image
normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
# Load the image into the array
data[0] = normalized_image_array

# run the inference
prediction = model.predict(data)
print(prediction)
```

## Milestone 3 - Making a RPS Game

- Made a manual RPS game where the user gives an input and is played against the computer. Random and regular expressions are the two modules used for this

```bash
conda activate rock
```

- The command above activates the conda environment with all the necessary libraries for the manual RPS game to function. The manual RPS game consists of 4 functions: get_user_choice, get_computer_choice, get_winner and play.

```python
import random
import re
```
```python
def get_computer_choice():
    choice_lst = ["rock", "paper", "scissors"]
    choice = random.choice(choice_lst)
    return choice
```
```python
def get_user_choice():
    choice_lst = ["rock", "paper", "scissors"]
    player_choice = None
    while player_choice not in choice_lst:
        player_choice = input("Please pick from Rock, Paper or Scissors: ").lower()
    return player_choice
```
```python
def get_winner(user_choice, computer_choice):
    if user_choice == computer_choice:
        print("tie! \n")
    if user_choice == "rock":
        if computer_choice == "paper":
            print("You Lose \n")
        elif computer_choice == "scissors":
            print("You Win! \n")
    if user_choice == "paper":
        if computer_choice == "scissors":
            print("You Lose \n")
        elif computer_choice == "rock":
            print("You Win! \n")
    if user_choice == "scissors":
        if computer_choice == "rock":
            print("You Lose \n")
        elif computer_choice == "paper":
            print("You Win! \n")
```
```python
def play():

    ynlst = ["y", "n"]
    while True:
        comp = get_computer_choice()
        user = get_user_choice()

        get_winner(user, comp)

        print("User:", user)
        print("Computer:", comp, "\n")

        

        play_again = None

        while play_again not in ynlst:
            play_again=input("Play again? (y/n): ").lower()

        if play_again != "y":
            break

    print("Bye!")
```

> Insert screenshot of what you have built working.

## Milestone 4 - Using the Camera to Play

The RPS game made in milestone 3 had to be adapted to take the data from the camera as the users choice. To do this the get_use_choice function was modified. The user is given 7 seconds or so to give enough data to be confident with the interpretation by the model. The models probabilities are counted and the class with the most amount of highest probabilities is taken as the users input. Then the game is played as normal.

```python
def get_user_choice():
    indexs = ["rock", "paper", "scissors", "nothing"]



    model = load_model('keras_model.h5')
    cap = cv2.VideoCapture(0)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    inds = []

    print("3")
    time.sleep(1)
    print("2")
    time.sleep(1)
    print("1")
    time.sleep(1)
    print("START!")

    t_end = time.time() + 60 * 0.2
    while time.time() < t_end: 
        ret, frame = cap.read()
        resized_frame = cv2.resize(frame, (224, 224), interpolation = cv2.INTER_AREA)
        image_np = np.array(resized_frame)
        normalized_image = (image_np.astype(np.float32) / 127.0) - 1 # Normalize the image
        data[0] = normalized_image
        prediction = model.predict(data)
        cv2.imshow('frame', frame)
        # Press q to close the window
        max = np.max(prediction)
        ind = np.argmax(prediction)
        inds.append(ind)
        choice = indexs[ind]
        #print(max, choice)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # After the loop release the cap object
    cap.release()
    # Destroy all the windows
    cv2.destroyAllWindows()

    values, counts = np.unique(inds, return_counts=True)
    choice_final = indexs[np.argmax(counts)]

    if len(values) == 1:
        if values[0] == 0:
            choice_final = "rock"
        if values[0] == 1:
            choice_final = "paper"
        if values[0] == 2:
            choice_final = "scissors"
        if values[0] == 3:
            choice_final = "nothing"

    return choice_final
```
The play function was also modified to include a 3 second countdown, aswell as a scoring algorithm to keep track of wins and losses.
```python

def play():
    rounds = 1
    user_score = 0
    user_loss = 0
    user_tie = 0
    comp_score = 0

    while rounds <= 5:

        print("ROUND", rounds)
        
        comp = get_computer_choice()
        user = get_user_choice()

        win = get_winner(user, comp)

        if win == 2:
            user_score += 1
        elif win == 1:
            comp_score += 1
            user_loss += 1
        elif win == 0:
            user_tie += 1


        print("User:", user)
        print("Computer:", comp, "\n")

        print("Wins =", user_score)

        if  win != 4:
            rounds += 1
        else:
            print("You didint give a valid input. Try again!")

    print("You won a total of:", user_score, "Games out of 5")
    print("Bye!")
```


## Conclusions

- This project helped me understand that breaking down the project into achievable goals can help understand whats going on and figure out what the next step is and how to do it. To improve, i would move the round timer onto the screen and make it so the camera doesnt close every round
