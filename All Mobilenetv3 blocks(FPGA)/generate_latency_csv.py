import csv
import os
import pandas as pd

if __name__ == "__main__":
    file_name = 'profile_summary.csv'

    all_latency_data = []
    blocks_list = os.listdir(os.getcwd())
    for block in blocks_list:
        if block.startswith('R'):
            latency_data =[]
            file_path = os.path.join(os.getcwd(), block,file_name)
            latency_data.append([block])
            if os.path.isfile(file_path):
                with open(file_path,'r') as file:
                    # latency_summary = pd.read_csv(file_path)
                    latency = csv.reader(file)
                    for row in latency:
                        latency_data.append(row)
            all_latency_data.append(latency_data)

    # file_path = 'output.txt'
    #
    # with open(file_path, 'w') as file:
    #     for item in all_latency_data:
    #         file.write(f"{item}\n")

    block_data = []
    count = 0
    for block in all_latency_data:
        for item in block:
            for i in item:
                if i.startswith('Residual'):
                    block_data.append([i])
                elif i.startswith('subgraph'):
                    block_data.append(item)

    df = pd.DataFrame(block_data, columns=latency_data[10])
    df.to_csv("latency_summary.csv")
    mask = df['Number Of Runs'].notnull()
    df.loc[mask, 'Kernel Name'] = df['Kernel Name'].shift(1)
    df_cleaned = df.dropna()
    # print(df_cleaned)
    df_cleaned.to_csv("cleaned_latency_summary.csv")
