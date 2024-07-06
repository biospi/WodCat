from pathlib import Path
import pandas as pd
import numpy as np


def main(input_folder):
    files = list(input_folder.glob("*.csv"))
    serial_list = []
    for file in files:
        #print(file)
        try:
            df = pd.read_csv(file, sep=",", header=None)
        except Exception as e:
            #print(e)
            print(file)
            continue

        data = df.head(15).values[:,0:2]
        for col, value in data:
            col = str(col)
            if 'serial' not in col.lower():
                continue
            serial = value
            #print(data)
            #print(f"Found serial {serial}")
            serial_list.append(serial)
    serial_list.append("Z146488") #Maisie
    print(np.unique(serial_list))


if __name__ == "__main__":
    input_folder = Path("E:\Cats")
    main(input_folder)