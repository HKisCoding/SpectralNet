import pandas as pd
import os 
import glob

from sklearn.preprocessing import LabelEncoder


PATH = 'dataset'
def create_dataset(name):
    data_dir = os.path.join(PATH, name)
    list_img = []
    if os.path.isdir(data_dir) == False:
        print (f"dataset {name} is not directory")
        raise Exception 
    
    for folder in glob.glob(data_dir + "\\*"):
        img_dir = os.path.join(data_dir, folder.split("\\")[-1])
        for item in glob.glob(img_dir + "\\*"):
            list_img.append({
                "img_name": item.split("\\")[-1],
                "label_name": folder.split("\\")[-1]
            })
    df = pd.DataFrame.from_dict(list_img)
    encoder = LabelEncoder()
    df['label'] = encoder.fit_transform(df['label_name'])
    return df


if __name__ == "__main__":
    name = "Caltech_101"
    # name = "MRSC-v2 Image Dataset"

    df = create_dataset(name)
    df.to_csv(f"{PATH}/{name}.csv")

