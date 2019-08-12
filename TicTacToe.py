from tkinter import *
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
import sys
from keras.models import model_from_json
import os
import time



class field_button:
    player = 1
    field = np.zeros((3,3))
    game_over = False
    current_choice = (None,None)
    ai_wrong_moves = 0

    def __init__(self,window,row,col):
        self.window = window
        self.row = row
        self.col = col
        self.button = Button(window,
                            text="",
                             bg=config["button_default_bg"],
                            command=self.clicked,
                             )
        self.button.config(width = 20,height = 10)
        self.button.grid(column=col, row=2*row+1)
        self.is_clicked = False

    def clicked(self):
        config["player"][field_button.player]["alias"] = e[field_button.player].get()
        player_dict = {1: player1_ai.get(),
                       -1: player2_ai.get()}
        print(field_button.field)
        print(field_button.player)
        field_button.current_choice = (self.row,self.col)
        if field_button.game_over:
            new_game_button_click()
        else:
            c = config["player"][field_button.player]["bg"]
            t = config["player"][field_button.player]["str"]

            if not self.is_clicked:
                self.button.configure(bg=c)
                self.button.configure(text=t)
                field_button.field[self.row, self.col] = field_button.player
                self.is_clicked = True
                check_var = self.check()
                if check_var == 0:
                    if np.sum(np.abs(field_button.field), axis=None) == 9:
                        new_game_button_click()
                    else:
                        field_button.player = -field_button.player
                        if player_dict[field_button.player] == 1:
                            ai_move()

            else:
                if player_dict[field_button.player]==1:
                    field_button.ai_wrong_moves += 1
                    if field_button.ai_wrong_moves > config["max_wrong_moves"]:
                        sys.exit(0)
                    wrong_move()
                    ai_move()


    def winner(self):
        if field_button.player == 1 :
            winner_alias = config["player"][1]["alias"]
        else:
            winner_alias = config["player"][-1]["alias"]
        self.button.configure(text=str("Winner: "+str(winner_alias)))
        if (field_button.player == 1 and player1_ai.get() == 1) or (field_button.player == -1 and player2_ai.get() == 1):
            good_move()
        else:
            bad_move()
        field_button.game_over = True

    def check(self):
        _field = np.array(field_button.field)
        ret = 0

        if np.any(np.abs(np.sum(_field,axis = 0)) == 3):
            self.winner()
            ret = 1

        if np.any(np.abs(np.sum(_field,axis = 1)) == 3):
            self.winner()
            ret = 1

        if np.abs(np.sum(np.eye(3)*_field,axis = None)) == 3:
            self.winner()
            ret = 1

        M=np.zeros((3,3))
        M[2, 0] = 1
        M[1, 1] = 1
        M[0, 2] = 1
        if np.abs(np.sum(M*_field,axis = None)) == 3:
            self.winner()
            ret = 1

        return ret

def ai_move():
    prediction = model.predict(field_button.field.flatten().reshape((1,9)))[0]
    index = np.argmax(prediction)
    print(prediction.reshape((3,3)))
    print("prediction: ",index,prediction[index])
    row = index//3
    col = index%3
    button_list[row][col].clicked()

def save_model(model):
    model.save_weights(config["weights"]+'.h5')

    # Save the model architecture
    with open(config["model_name"]+'.json', 'w') as f:
        f.write(model.to_json())

def good_move():
    print("good move")
    data = field_button.player*field_button.field.flatten()
    target = np.zeros(9)
    row,col = field_button.current_choice
    index=3*row+col
    target[index] = 1
    model.fit(data.reshape((1,9)), target.reshape((1,9)), epochs=config["n_training_epochs"], verbose=False)
    save_model(model)

def bad_move():
    print("bad move")
    data = -field_button.player*field_button.field.flatten()
    target = np.zeros(9)
    row,col = field_button.current_choice
    index=3*row+col
    target[index] = 1
    model.fit(data.reshape((1,9)), target.reshape((1,9)), epochs=config["n_training_epochs"], verbose=False)
    save_model(model)

def wrong_move():
    data = field_button.player*field_button.field.flatten()
    target = np.ones(9)-np.abs(field_button.field.flatten())
    target = target/max(np.sum(target),1)
    print("wrong move")
    model.fit(data.reshape((1, 9)), target.reshape((1, 9)), epochs=config["n_training_epochs"], verbose=False)
    model.fit(-data.reshape((1, 9)), target.reshape((1, 9)), epochs=config["n_training_epochs"], verbose=False)
    save_model(model)

def new_game_button_click():
    field_button.player = 1
    field_button.field = np.zeros((3, 3))
    for row in range(3):
        for col in range(3):
            b=button_list[row][col]
            b.is_clicked = False
            b.game_over = False
            b.button.configure(bg=config["button_default_bg"])
            b.button.configure(text="")
            field_button.game_over = False
    if player1_ai.get() == 1:
        ai_move()

def make_model():
    if e["model_name"].get() != "":
        config["model_name"] = e["model_name"].get()

    config["weights"] = config["model_name"] + "_weights"
    if os.path.isfile(config["model_name"]+".json"):
        # Model reconstruction from JSON file
        with open(config["model_name"]+'.json', 'r') as f:
            model = model_from_json(f.read())

        # Load weights into the new model
        model.load_weights(config["weights"]+'.h5')
        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        print(config["model_name"]+"model loaded")

    else:
        # AI PLAYER IS -1
        model = Sequential([
            Dense(30, input_shape=(9,)),
            Activation('relu'),
            Dense(60),
            Activation('relu'),
            Dense(40),
            Activation('relu'),
            Dense(20),
            Activation('relu'),
            Dense(9),
            Activation('softmax'),
        ])

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

config = {
    "windows_title": "Tic Tac Toe",
    "window_geometrie": '800x800',
    "button_default_bg": "grey",
    "max_wrong_moves":1000,
    "n_training_epochs":200,
    "model_name":"model",
    "weights":"weights",
    "player":{1: {"bg":"orange",
                  "alias": "Orange",
                  "str":"X"
                  },
              -1 : {"bg":"blue",
                  "alias": "Blue",
                   "str":"O"
                  }
              }
}

config["weights"]=config["model_name"]+"_weights"

window = Tk()
window.geometry(config["window_geometrie"])

lbl = Label(window, text=config["windows_title"])
lbl.grid(column=1, row=0)

button_list = [[0 for i in range(7)]for i in range(7)]

for row in range(3):
    for col in range(3):
        button_list[row][col] = field_button(window,row,col)

new_game_button = Button(window,
                        text="new game",
                        command=new_game_button_click)
new_game_button.grid(column=1, row=8)

player1_ai = IntVar()
ch_1 = Checkbutton(window, text="player 1 AI", variable=player1_ai, bg = "orange").grid(row=8,column =0)
player2_ai = IntVar()
ch_2 = Checkbutton(window, text="player 2 AI", variable=player2_ai, bg = "blue",).grid(row=8,column =2)

e={}
e[1] = Entry(window)
e[1] .grid(row = 9, column =0)
e[1] .delete(0, END)
e[1] .insert(0, "Orange")

e[-1] = Entry(window)
e[-1].grid(row = 9, column =2)
e[-1].delete(0, END)
e[-1].insert(0, "Blue")

e["model_name"] = Entry(window)
e["model_name"].grid(row = 10, column = 1)
e["model_name"].delete(0, END)
e["model_name"].insert(0, "")

model_button = Button(window,
                        text="make model",
                        command=make_model)
model_button.grid(column=1, row=11)

model = make_model()

window.mainloop()




