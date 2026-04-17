from sklearn.datasets import fetch_openml
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import numpy as np

# load quantized input scaling factor
S_input = np.load("data/input_scalingfactor.npy")

# prepare data 
penguins = fetch_openml(name = 'penguins', parser = "auto", as_frame = True).frame
penguins = penguins.dropna(axis=0) # remove all rows(axis=0) that are missing values

i = penguins[['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']]
i = MinMaxScaler().fit_transform(i) # scale measurements to [0, 1]
t = penguins['species']
t = LabelEncoder().fit_transform(t) # encode target as a number


q_input = np.round(i / S_input)
q_input = np.clip(q_input, -128, 127).astype(np.int8) # limit quantized weights to range -127, 127 and convert to 8-bit-int

# write testvector
with open("neurontest.dat", "w") as f:
    for i in range(len(q_input)):
        inp = q_input[i].tolist()
        inp.append(t[i].item())
        line = " ".join(str(v) for v in inp)
        f.write(line + "\n")