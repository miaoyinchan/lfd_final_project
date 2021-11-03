import os
import shutil

SRC = "Saved Models/"
DES = "experiments/"

def main():
    folders = os.listdir(SRC)
    for f in folders:
        
        shutil.move(SRC+f+"/Saved_Models", DES+f)

        
    shutil.rmtree(SRC)


if __name__ == "__main__":
    main()