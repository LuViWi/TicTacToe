from tkinter import *
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
import sys
from keras.models import model_from_json
import os
import numpy as np
import sys

def feld_anzeigen(feld):
    print(feld)

def gültiges_feld(s, feld):
    # s ist die input spalte typ int
    for i in [6, 5, 4, 3, 2, 1, 0]:
        if feld[i][int(s)] == int(0):
            return i
    print("die spalte",s,"ist voll")
    return -1


def move(player, s, feld,data=True):
    z = gültiges_feld(s,feld)
    if z!= -1:
        feld[z][s] = int(player)
        #    if data:
        #        make_data(z,s,player,0)
    else:
        z = -1
    return z



def check_horizontal(feld, resque=False):
    h1 = np.array([1, 1, 1, 1, 0, 0, 0])
    h2 = np.array([0, 1, 1, 1, 1, 0, 0])
    h3 = np.array([0, 0, 1, 1, 1, 1, 0])
    h4 = np.array([0, 0, 0, 1, 1, 1, 1])

    m = np.array([h1, h2, h3, h4])

    for v in m:
        for b in feld.dot(v):
            if abs(b) == 4:
                return np.sign(b)
    return 0

def check_vertical(feld, resque=False):
    h1 = np.array([1, 1, 1, 1, 0, 0, 0])
    h2 = np.array([0, 1, 1, 1, 1, 0, 0])
    h3 = np.array([0, 0, 1, 1, 1, 1, 0])
    h4 = np.array([0, 0, 0, 1, 1, 1, 1])

    m = np.array([h1, h2, h3, h4])
    e = np.identity(7)

    for v in m:
        for b in feld.T.dot(v):
            if abs(b) == 4:
                return np.sign(b)
    return 0

def check_diagonal(feld):
    h1 = np.array([1, 1, 1, 1, 0, 0, 0])
    h2 = np.array([0, 1, 1, 1, 1, 0, 0])
    h3 = np.array([0, 0, 1, 1, 1, 1, 0])
    h4 = np.array([0, 0, 0, 1, 1, 1, 1])
    m = np.array([h1, h2, h3, h4])
    spiegel = np.array([[0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0]])
    for l in range(len(m)):
        n=m[l]
        for j in range(2):  # 2 richtungen zum shiften =  oben rechts und untel links
            mat = np.diag(n)
            for i in range(4-l):  # 4 mal shiften pro richtung
                if abs(sum(sum(mat * feld))) == 4:  # test diagonale
                    return np.sign(sum(sum(mat * feld)))

                mat2 = spiegel.dot(mat)  # horizontzal spiegeloperation
                if abs(sum(sum(mat2 * feld))) == 4:
                    return np.sign(sum(sum(mat2 * feld)))

                mat = np.insert(mat, 0, 0, j)  # shiften
                mat = np.delete(mat, 7, j)
    return 0

def check(feld):
    cv = check_vertical(feld)
    ch = check_horizontal(feld)
    cd = check_diagonal(feld)

    if not cv == 0:
        return cv
    if not ch == 0:
        return ch
    if not cd == 0:
        return cd

    return 0


def pvp():
    feld = np.zeros((7, 7))
    feld_anzeigen(feld)
    current_player = 1

    while 1:
        print("Spieler", current_player, "ist dran")
        s = int(input("Welche Spalte? (0-6) / 99 für beenden"))

        if s == int(99):
            break

        if not s in range(7):
            print("keine gültige zahl")
            continue

        move(current_player, s, feld)
        feld_anzeigen(feld)
        current_player *= -1

        check_it = check(feld)

        if not check_it == int(0):
            print("Player", check_it, "won")
            break

    #make_data(0, 0, 0, check_it)
    print("ende")



def save_model(model):
    model.save_weights(config["weights"]+'.h5')

    # Save the model architecture
    with open(config["model_name"]+'.json', 'w') as f:
        f.write(model.to_json())


def good_move():
    print("good move")
    data = field_button.player*field_button.field.flatten()
    target = np.zeros(7)
    index = field_button.current_choice
    target[index] = 1
    model.fit(data.reshape((1,49)), target.reshape((1,7)), epochs=config["n_training_epochs"], verbose=False)
    save_model(model)
    make_data(data.reshape((1,49)),target.reshape((1,7)))

def bad_move():
    print("bad move")
    data = -field_button.player*field_button.field.flatten()
    target = np.zeros(7)
    index = field_button.current_choice
    target[index] = 1
    model.fit(data.reshape((1,49)), target.reshape((1,7)), epochs=config["n_training_epochs"], verbose=False)
    save_model(model)
    make_data(data.reshape((1, 49)), target.reshape((1, 7)))

def wrong_move():
    data = field_button.player*field_button.field.flatten()
    target = np.ones(7)
    target[field_button.current_choice] = 0
    target = target/max(np.sum(target),1)
    print("wrong move")
    model.fit(data.reshape((1, 49)), target.reshape((1, 7)), epochs=config["n_training_epochs"], verbose=False)
    model.fit(-data.reshape((1, 49)), target.reshape((1, 7)), epochs=config["n_training_epochs"], verbose=False)
    save_model(model)




config = {
    "windows_title": "Connect 4",
    "window_geometrie": '800x800',
    "button_default_bg": "green",
    "max_wrong_moves":1000,
    "n_training_epochs":200,
    "model_name":"connect4_model",
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



def new_game_button_click():
    field_button.player = 1
    field_button.field = np.zeros((7, 7))
    for row in range(7):
        for col in range(7):
            b=button_list[row][col]
            b.is_clicked = False
            b.configure(bg="gray")
            b.configure(text="")
            field_button.game_over = False

    if player1_ai.get() == 1:
        ai_move()

def ai_move():
    prediction = model.predict(field_button.field.flatten().reshape((1,49)))[0]
    index = np.argmax(prediction)
    print("prediction: ",index,prediction[index])
    control_button_list[index].clicked()


class field_button:
    player = 1
    field = np.zeros((7,7))
    game_over = False
    current_choice = (None,None)
    ai_wrong_moves = 0

    def __init__(self,window,row,col):
        self.index = col
        self.window = window
        self.row = row
        self.col = col
        self.button = Button(window,
                            text="",
                             bg=config["button_default_bg"],
                            command=self.clicked,
                             )
        self.button.config(width = 5,height = 5)
        self.button.grid(column=col, row=row+1)
        self.is_clicked = False

    def clicked(self):
        print(field_button.player)
        print(field_button.field)
        ai_player_dict = {1: player1_ai.get(),
                       -1: player2_ai.get()}
        c = config["player"][field_button.player]["bg"]
        t = config["player"][field_button.player]["str"]
        row = gültiges_feld(self.index,field_button.field)
        field_button.current_choice = self.index
        if not field_button.game_over:
            if row != -1:
                field_button.ai_wrong_moves = 0
                pre_field = np.array(field_button.field)
                field_button.field[row,self.index]=field_button.player
                button_list[row][self.index].config(bg=c)

                print(field_button.player, player1_ai.get(), player2_ai.get())

                check_var = check(field_button.field)
                if check_var == 0:
                    if np.sum(np.abs(field_button.field), axis=None) == 49:
                        new_game_button_click()
                    else:
                        field_button.player = -field_button.player
                        if ai_player_dict[field_button.player] == 1:
                            ai_move()
                else:
                    button_list[row][self.index].config(text="Winner "+config ["player"][field_button.player]["alias"])
                    if (field_button.player == 1 and player1_ai.get() == 1):
                        good_move()
                    if (field_button.player == -1 and player2_ai.get() == 1):
                        good_move()
                    if (field_button.player == 1 and player2_ai.get() == 1):
                        bad_move()
                    if (field_button.player == -1 and player1_ai.get() == 1):
                        bad_move()
                    field_button.game_over=True


            else:
                if ai_player_dict[field_button.player] == 1:
                    field_button.ai_wrong_moves += 1
                if field_button.ai_wrong_moves > config["max_wrong_moves"]:
                    sys.exit(0)
                wrong_move()
                ai_move()
        else:
            new_game_button_click()

def make_data(data,target):

    if os.path.isfile(config["model_name"]+"_data.npy"):
        old_data=np.load(config["model_name"]+"_data.npy","r")
    else:
        old_data = np.zeros(49)
    if os.path.isfile(config["model_name"]+"_target.npy"):
        old_target=np.load(config["model_name"]+"_target.npy","r")
    else:
        old_target = np.zeros(7)
    print(len(old_target))
    np.save(config["model_name"]+"_data.npy",np.vstack((data,old_data)))
    np.save(config["model_name"] + "_target.npy", np.vstack((target, old_target)))

def train_data():
    if os.path.isfile(config["model_name"]+"_data.npy"):
        data=np.load(config["model_name"]+"_data.npy","r")
    else:
        return 0
    if os.path.isfile(config["model_name"]+"_target.npy"):
        target=np.load(config["model_name"]+"_target.npy","r")
    else:
        return 0
    model.fit(data, target, epochs=config["n_training_epochs"], verbose=False)
    print("model "+config["model_name"]+" trained")

def make_model():
    if e["model_name"].get() != "your_model_name":
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
            Dense(70, input_shape=(49,)),
            Activation('relu'),
            Dense(60),
            Activation('relu'),
            Dense(40),
            Activation('relu'),
            Dense(20),
            Activation('relu'),
            Dense(7),
            Activation('softmax'),
        ])
        print("model "+config["model_name"]+" made")
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


window = Tk()
window.geometry(config["window_geometrie"])

lbl = Label(window, text=config["windows_title"])
lbl.grid(column=3, row=0)

button_list = [[0 for i in range(7)]for i in range(7)]

for row in range(7):
    for col in range(7):
        button_list[row][col] =Button(window,
                            text="",
                             bg="gray",
                            command=None,
                             )
        button_list[row][col].config(width = 8,height = 4)
        button_list[row][col].grid(column=col, row=row+1)

control_button_list = [0 for i in range(7)]
for col in range(7):
    control_button_list[col] = field_button(window,9,col)

player1_ai = IntVar()
ch_1 = Checkbutton(window, text="player 1 AI", variable=player1_ai, bg = "orange")
ch_1.grid(row=11,column =2)
player2_ai = IntVar()
ch_2 = Checkbutton(window, text="player 2 AI", variable=player2_ai, bg = "blue",)
ch_2.grid(row=11,column =4)

new_game_button = Button(window,
                        text="new game",
                        command= new_game_button_click)
new_game_button.grid(column=3, row=12)

e={}
e[1] = Entry(window, width = 5)
e[1] .grid(row = 12, column =1)
e[1] .delete(0, END)
e[1] .insert(0, "Orange")

e[-1] = Entry(window, width = 5)
e[-1].grid(row = 12, column =5)
e[-1].delete(0, END)
e[-1].insert(0, "Blue")

e["model_name"] = Entry(window, width = 10)
e["model_name"].grid(row = 14, column = 3)
e["model_name"].delete(0, END)
e["model_name"].insert(0, "your_model_name")

model_button = Button(window,
                        text="make model",
                        command=make_model)
model_button.grid(column=1, row=14)

train_button = Button(window,
                        text="train model",
                        command=train_data)
train_button.grid(column=5, row=14)
model = make_model()
window.mainloop()



