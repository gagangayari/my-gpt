from datasets import load_dataset
dataset = load_dataset("roneneldan/TinyStories", split= "train[:40%]")

# print(dataset[0])
with open("tinyStories.txt", 'w') as file:
    for i,row in enumerate(dataset):
        file.write(row['text'])
        file.write("\n")

        if (i == 10000):
            break




