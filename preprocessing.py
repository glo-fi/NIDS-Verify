from ast import literal_eval
import pandas as pd
import numpy as np
import tensorflow as tf
import os




# Data Loading

def load_meta_pkt_df(input_dir, output_dir=None, save=False):
    print(f"[*] Loading Dataset From {input_dir}")
    pkt_df = pd.DataFrame()
    meta_df = pd.DataFrame()
    temp_df = pd.DataFrame()
    for sub_dir in os.listdir(input_dir):
        d = os.path.join(input_dir, sub_dir)
        if os.path.isdir(d):
            for f in os.listdir(d):
                if "csv" in f:
                    name = f.split('.')[0].split('-')[-1]
                    if name == 'metadata':
                        temp_df = pd.read_csv(os.path.join(d, f), sep=',', header=None)
                        temp_df.columns = ['Timestamp', 'Unix Timestamp', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Protocol']
                        meta_df = pd.concat([meta_df, temp_df])
                    else:
                        temp_df = pd.read_csv(os.path.join(d, f), delimiter='\t', header=None)
                        temp_df.columns = [name]
                        pkt_df = pd.concat([pkt_df, temp_df], axis=1)
    if save:
            meta_df.to_csv(f'./{output_dir}/metadata.csv', index=True)
            pkt_df.to_csv(f'./{output_dir}/pkt.csv', index=True)

    return meta_df, pkt_df

def load_nested_meta_pkt_df(input_dir, output_dir=None, save=False):
    print(f"[*] Loading Dataset From {input_dir}")
    pkt_df = pd.DataFrame()
    meta_df = pd.DataFrame()
    temp_df = pd.DataFrame()
    for sub_dir in os.listdir(input_dir):
        d1 = os.path.join(input_dir, sub_dir)
        for sub_sub_dir in os.listdir(d1):
            d2 = os.path.join(d1, sub_sub_dir)
            temp_pkt_df = pd.DataFrame()
            if os.path.isdir(d2):
                for f in os.listdir(d2):
                    if "csv" in f:
                        name = f.split('.')[0].split('-')[-1]
                        if name == 'metadata':
                            temp_df = pd.read_csv(os.path.join(d2, f), sep=',', header=None)
                            temp_df.columns = ['Timestamp', 'Unix Timestamp', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Protocol']
                            meta_df = pd.concat([meta_df, temp_df])
                        else:
                            temp_df = pd.read_csv(os.path.join(d2, f), delimiter='\t', header=None)
                            temp_df.columns = [name]
                            temp_pkt_df = pd.concat([temp_pkt_df, temp_df], axis=1)
            pkt_df = pd.concat([pkt_df, temp_pkt_df])
    if save:
        meta_df.to_csv(f'./{output_dir}/metadata.csv', index=True)
        pkt_df.to_csv(f'./{output_dir}/pkt.csv', index=True)

    return meta_df, pkt_df

# Merge the columns of the metadata and packet dataframes
def combine_meta_pkt_dfs(meta_df, pkt_df):
    return pd.concat([meta_df, pkt_df], axis=1)


def get_datasets(train_pos, train_neg, test_pos, test_neg):
    if train_pos == "CIC2017":
        pos_train_dir = './data/CICWednesdayData/pos_neg/pos2_wednesday.csv'
        pos_train_df = pd.read_csv(pos_train_dir)
    elif train_pos == "CIC2018":
        pos_train_dir = './data/CIC2018Data/'
        meta_pos_train_df, pkt_pos_train_df = load_meta_pkt_df(pos_train_dir)
        pos_train_df = combine_meta_pkt_dfs(meta_pos_train_df, pkt_pos_train_df)

    if train_neg == "CIC2017":
        neg_train_dir = './data/CICWednesdayData/pos_neg/neg2_wednesday.csv' # All DoS Attacks
        neg_train_df = pd.read_csv(neg_train_dir)
    elif train_neg == "CIC2018":
        neg_train_dir = './data/CIC2018Hulk/' # Just Hulk
        meta_neg_train_df, pkt_neg_train_df = load_meta_pkt_df(neg_train_dir)
        neg_train_df = combine_meta_pkt_dfs(meta_neg_train_df, pkt_neg_train_df)
    elif train_neg == "DetGen":
        neg_train_dir = './data/DoSData/http_flood/25mb/' # (Just HTTPFlood (basically Hulk but technically not really))
        meta_neg_train_df, pkt_neg_train_df = load_nested_meta_pkt_df(neg_train_dir)
        neg_train_df = combine_meta_pkt_dfs(meta_neg_train_df, pkt_neg_train_df)
    elif train_neg == "CIC2017SSH":
        neg_train_dir = './data/CICTuesdayData/pos_neg/ssh.csv' 
        neg_train_df = pd.read_csv(neg_train_dir)
    elif train_neg == "CIC2017FTP":
        neg_train_dir = './data/CICTuesdayData/pos_neg/ftp.csv' 
        neg_train_df = pd.read_csv(neg_train_dir)

    if test_pos == "CIC2017":
        pos_test_dir = './data/CICWednesdayData/pos_neg/pos_wednesday.csv'
        pos_test_df = pd.read_csv(pos_test_dir)
    elif test_pos == "CIC2018":
        pos_test_dir = './data/CIC2018Data/'
        meta_pos_test_df, pkt_pos_test_df = load_meta_pkt_df(pos_test_dir)
        pos_test_df = combine_meta_pkt_dfs(meta_pos_test_df, pkt_pos_test_df)

    if test_neg == "CIC2017":
        neg_test_dir = './data/CICWednesdayData/pos_neg/neg_wednesday.csv'
        neg_test_df = pd.read_csv(neg_test_dir)
    elif test_neg == "CIC2018":
        neg_test_dir = './data/CIC2018Hulk/'
        meta_neg_test_df, pkt_neg_test_df = load_meta_pkt_df(neg_test_dir)
        neg_test_df = combine_meta_pkt_dfs(meta_neg_test_df, pkt_neg_test_df)
    elif test_neg == "DetGen":
        neg_test_dir = './data/DoSData/http_flood/25mb/'
        meta_neg_test_df, pkt_neg_test_df = load_nested_meta_pkt_df(neg_test_dir)
        neg_test_df = combine_meta_pkt_dfs(meta_neg_test_df, pkt_neg_test_df)
    elif test_neg == "CIC2018SSH":
        neg_test_dir = './data/CIC2018Hulk/'
        neg_test_df = pd.read_csv(neg_test_dir)
    elif test_neg == "DetGenSSH":
        neg_test_dir = './data/DetGenSSH/'
        meta_neg_test_df, pkt_neg_test_df = load_meta_pkt_df(neg_test_dir)
        neg_test_df = combine_meta_pkt_dfs(meta_neg_test_df, pkt_neg_test_df)
    elif test_neg == "CIC2017FTP": # Using FTP Bruteforce for now, just because
        neg_test_dir = './data/CICTuesdayData/pos_neg/ftp.csv' 
        neg_test_df = pd.read_csv(neg_test_dir)

    return pos_train_df, neg_train_df, pos_test_df, neg_test_df

def preprocess(df):
    print(f"\t[*] Preprocessing . . .")
    # Sort the DataFrame by 'Source IP', 'Destination IP', and 'Unix Timestamp'
    df.sort_values(by=['Source IP', 'Destination IP', 'Unix Timestamp'], inplace=True)
    # Calculate the time elapsed for each group of rows
    df['Time_Elapsed'] = df.sort_values(by=['Unix Timestamp']).groupby(['Source IP', 'Destination IP'])['Unix Timestamp'].diff()
    # Replace NaN values in 'Time_Elapsed' with 0 (for the first row in each group)
    df['Time_Elapsed'].fillna(0, inplace=True)
    # Replace 6 with 0 and 17 with 1 in the specified column
    df['Protocol'] = df['Protocol'].replace({6: 0, 17: 1})
    # Convert strings to lists
    df[['Pkt_Direction', 'Pkt_Flags', 'Pkt_IATs', 'Pkt_Sizes']] = df[['Pkt_Direction', 'Pkt_Flags', 'Pkt_IATs', 'Pkt_Sizes']].applymap(literal_eval)
    # Drop unwanted columns
    df = df[['Time_Elapsed', 'Protocol', 'Pkt_Direction', 'Pkt_Flags', 'Pkt_IATs', 'Pkt_Sizes']]
    return df


def load_datasets(train_pos, train_neg, test_pos, test_neg, resample=None, balance=True):
    pos_train_df, neg_train_df, pos_test_df, neg_test_df = get_datasets(train_pos, train_neg, test_pos, test_neg)

    print("\t[*] Rebalancing data . . .")
    # Drop unwanted rows
    neg_test_df = neg_test_df.drop(neg_test_df[neg_test_df['Protocol'] == 17].index) # Dropped accidental UDP packets
    if resample != None:
        neg_test_df = neg_test_df.sample(frac=resample)
    neg_test_df = neg_test_df.reset_index(drop=True)

    # Randomly sample from the negative dataframe to match the size of the positive dataframe
    # This probably isn't best practise but it's also what LUCID does (our main comparison), so let's leave it in
    if balance:
        pos_train_df = pos_train_df.sample(n=len(neg_train_df.index), random_state=42)

        pos_test_df = pos_test_df.sample(n=len(neg_test_df.index), random_state=42)
    pos_train_df = preprocess(pos_train_df)
    print(f"POS TRAIN SHAPE: {pos_train_df.shape}")
    neg_train_df = preprocess(neg_train_df)
    print(f"NEG TRAIN SHAPE: {neg_train_df.shape}")

    pos_test_df = preprocess(pos_test_df)
    print(f"POS TEST SHAPE: {pos_test_df.shape}")
    neg_test_df = preprocess(neg_test_df)
    print(f"NEG TEST SHAPE: {neg_test_df.shape}")

    return pos_train_df, neg_train_df, pos_test_df, neg_test_df


def _feature_engineer(df, m=4, columns_to_transform=['Pkt_Direction', 'Pkt_IATs', 'Pkt_Sizes', 'Pkt_Flags'], save=False):
    print(f"\t[*] Feature engineering with packet depth {m}")
    for col in columns_to_transform:
        df[col] = df[col].apply(lambda x: transform_row_from_tuple(list(x), m) if isinstance(x, (list, tuple)) else transform_row_from_tuple([x], m))
        
    # Custom function to flatten lists and lists of lists
    def flatten_element(cell):
        if isinstance(cell, list):
            return [item for sublist in cell for item in sublist] if any(isinstance(item, list) for item in cell) else cell
        return cell

    df = df.applymap(flatten_element)
    return df


def feature_engineer(pos_train_df, neg_train_df, pos_test_df, neg_test_df, m):
    pos_train_df = _feature_engineer(pos_train_df, m=m)
    neg_train_df = _feature_engineer(neg_train_df, m=m)
    pos_test_df = _feature_engineer(pos_test_df, m=m)
    neg_test_df = _feature_engineer(neg_test_df, m=m)

    return pos_train_df, neg_train_df, pos_test_df, neg_test_df


def _to_tensorflow(pos_df, neg_df, time_max=50000000000, size_max=1000, iat_max=5000000000, flag_max=256, save=False, train=True, attack_name="DoS"):
    print("\t[*] Converting to Tensorflow format")
    # Add the labels column
    pos_df['label'] = 0
    pos_df['Time_Elapsed'] = pos_df['Time_Elapsed'].apply(scale_column, min=0, max=time_max)
    neg_df['label'] = 1
    neg_df['Time_Elapsed'] = neg_df['Time_Elapsed'].apply(scale_column, min=0, max=time_max)

    # Concatenate the 2 dataframes
    data_df = pd.concat([pos_df, neg_df])

    # Save the dataframe
    if save:
        data_df.to_csv(f'./data/data_{attack_name.lower()}.csv', index=True)

    # Convert 'Pkt_Direction,' 'Pkt_Flags,' 'Pkt_IATs,' and 'Pkt_Sizes' to lists
    data_df['Pkt_Direction'] = data_df['Pkt_Direction'].apply(list)
    data_df['Pkt_Flags'] = data_df['Pkt_Flags'].apply(list)
    data_df['Pkt_IATs'] = data_df['Pkt_IATs'].apply(list)
    data_df['Pkt_Sizes'] = data_df['Pkt_Sizes'].apply(list)

    # Ensure consistent data types (float32 for floating-point values, int32 for integers)
    # data_df['Time_Elapsed'] = data_df['Time_Elapsed'].astype('int32')                                             # This makes it negative, for some reason
    data_df['Protocol'] = data_df['Protocol'].astype('int32')
    data_df['label'] = data_df['label'].astype('int32')
        
    # Normalise columns
    data_df['Pkt_Sizes'] = data_df['Pkt_Sizes'].apply(scale_column, min=0, max=size_max)                                # GLOBAL VARIABLES (NEEDS TO BE LOOKED AT)
    data_df['Pkt_IATs'] = data_df['Pkt_IATs'].apply(scale_column, min=0, max=iat_max)                                  # GLOBAL VARIABLES (NEEDS TO BE LOOKED AT)
    #data_df['Time_Elapsed'] = data_df['Time_Elapsed'].apply(scale_column, min=0, max=500000000) #max=500000000000)                         # GLOBAL VARIABLES (NEEDS TO BE LOOKED AT)
    data_df['Pkt_Flags'] = data_df['Pkt_Flags'].apply(scale_column, min=0, max=flag_max)                                # GLOBAL VARIABLES (NEEDS TO BE LOOKED AT)

    if save:
        # Save the dataframe
        data_df.to_csv(f'./data/data_{attack_name.lower()}.csv', index=False)

    numpy_df = data_df.to_numpy()
    new_numpy_df = []
    for i in range(numpy_df.shape[0]):
        # Flatten the numpy list into a single list
        flattened_list = [item for sublist in numpy_df[i] for item in (sublist if isinstance(sublist, list) else [sublist])]
        new_numpy_df.append(flattened_list)
    new_numpy_df = np.array(new_numpy_df)

    new_df = pd.DataFrame(new_numpy_df)
    if save:
        # Save the dataframe
        if train:
            new_df.to_csv(f'./data/full-preprocessed-{attack_name.lower()}-data-train.csv', index=False)
        elif not train:
            new_df.to_csv(f'./data/full-preprocessed-{attack_name.lower()}-data-test.csv', index=False)

    features = new_df.iloc[:, :(len(new_df.columns)-1)]
    labels = new_df.iloc[:, -1:]

    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    return dataset, data_df, new_df


def to_tensorflow(pos_train_df, neg_train_df, pos_test_df, neg_test_df, time_max, size_max, iat_max, flag_max, batch_size=2048, save=False, attack_name="DoS"):
    train_dataset, _, _ = _to_tensorflow(pos_train_df,
                                         neg_train_df,
                                         time_max,
                                         size_max,
                                         iat_max,
                                         flag_max,
                                         save,
                                         train=True,
                                         attack_name=attack_name)

    test_dataset, _, _ = _to_tensorflow(pos_test_df,
                                        neg_test_df,
                                        time_max,
                                        size_max,
                                        iat_max,
                                        flag_max,
                                        save,
                                        train=False,
                                        attack_name=attack_name)  

    print(f"\t[*] Batch size: {batch_size}")
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
    test_dataset = test_dataset.batch(batch_size)
    return train_dataset, test_dataset  


def pipeline(pos_train,
            neg_train,
            pos_test,
            neg_test,
            resample=None,
            m=10,
            batch_size=2048,
            preprocess_dict={"time_max": 50000000000,
                             "iat_max": 5000000000,
                             "size_max": 1000,
                             "flag_max": 256},
            save=False,
            load=False,
            attack_name="DoS"):
    if load:
        train_data_df = pd.read_csv(f'./data/preprocessed-dos-train.csv')
        train_features = train_data_df.iloc[:, :(len(train_data_df.columns)-1)]
        train_labels = train_data_df.iloc[:, -1:]
        print(train_labels.value_counts())


        test_data_df = pd.read_csv(f'./data/preprocessed-dos-test.csv')
        print(test_data_df.columns)
        test_data_df = test_data_df[test_data_df['42'] == 1]
        test_features = test_data_df.iloc[:, :(len(test_data_df.columns)-1)]
        test_labels = test_data_df.iloc[:, -1:]


        train_dataset = tf.data.Dataset.from_tensor_slices((train_features, train_labels))
        test_dataset = tf.data.Dataset.from_tensor_slices((test_features, test_labels))

        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
        test_dataset = test_dataset.batch(batch_size)

        return train_dataset, test_dataset

    else:
        print("[!!] Beginning Pipeline")
        pos_train_df, neg_train_df, pos_test_df, neg_test_df = load_datasets(pos_train, 
                                                                            neg_train, 
                                                                            pos_test, 
                                                                            neg_test,
                                                                            resample=resample)

        pos_train_df, neg_train_df, pos_test_df, neg_test_df = feature_engineer(pos_train_df, 
                                                                                neg_train_df, 
                                                                                pos_test_df, 
                                                                                neg_test_df, 
                                                                                m=m)

        train_dataset, test_dataset = to_tensorflow(pos_train_df,
                                                    neg_train_df,
                                                    pos_test_df,
                                                    neg_test_df,
                                                    preprocess_dict["time_max"],
                                                    preprocess_dict["size_max"],
                                                    preprocess_dict["iat_max"],
                                                    preprocess_dict["flag_max"],
                                                    batch_size=batch_size,
                                                    save=save,
                                                    attack_name=attack_name)

        return train_dataset, None, test_dataset

def preprocess_dataset(pos_dir, neg_dir, length, save_path):

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Bening data
    pkt_df_pos = pd.DataFrame()
    meta_df_pos = pd.DataFrame()
    for sub_dir in os.listdir(pos_dir):
        d = os.path.join(pos_dir, sub_dir)
        if os.path.isdir(d):
            for f in os.listdir(d):
                name = f.split('.')[0].split('-')[-1]
                if name == 'metadata':
                    temp_df_pos = pd.read_csv(os.path.join(d, f), sep=',', header=None)
                    temp_df_pos.columns = ['Timestamp', 'Unix Timestamp', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Protocol']
                    meta_df_pos = pd.concat([meta_df_pos, temp_df_pos])
                else:
                    temp_df_pos = pd.read_csv(os.path.join(d, f), delimiter='\t', header=None)
                    temp_df_pos.columns = [name]
                    pkt_df_pos = pd.concat([pkt_df_pos, temp_df_pos], axis=1)

    # Malicious data
    pkt_df_neg = pd.DataFrame()
    meta_df_neg = pd.DataFrame()
    for sub_dir in os.listdir(neg_dir):
        d = os.path.join(neg_dir, sub_dir)
        temp_pkt_df_neg = pd.DataFrame()
        if os.path.isdir(d):
            for f in os.listdir(d):
                name = f.split('.')[0].split('-')[-1]
                if name == 'metadata':
                    temp_df_neg = pd.read_csv(os.path.join(d, f), sep=',', header=None)
                    temp_df_neg.columns = ['Timestamp', 'Unix Timestamp', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Protocol']
                    meta_df_neg = pd.concat([meta_df_neg, temp_df_neg])
                else:
                    temp_df_neg = pd.read_csv(os.path.join(d, f), delimiter='\t', header=None)
                    temp_df_neg.columns = [name]
                    temp_pkt_df_neg = pd.concat([temp_pkt_df_neg, temp_df_neg], axis=1)
        pkt_df_neg = pd.concat([pkt_df_neg, temp_pkt_df_neg])

    # Save the meta and pkt dataframes
    meta_df_pos.to_csv(f'{save_path}/metadata-pos.csv', index=True)
    meta_df_neg.to_csv(f'{save_path}/metadata-neg.csv', index=True)
    pkt_df_pos.to_csv(f'{save_path}/pkt-pos.csv', index=True)
    pkt_df_neg.to_csv(f'{save_path}/pkt-neg.csv', index=True)

    # Load the meta and pkt dataframes
    meta_df_pos = pd.read_csv(f'{save_path}/metadata-pos.csv')
    meta_df_neg = pd.read_csv(f'{save_path}/metadata-neg.csv')
    pkt_df_pos = pd.read_csv(f'{save_path}/pkt-pos.csv')
    pkt_df_neg = pd.read_csv(f'{save_path}/pkt-neg.csv')

    # Merge the columns of the metadata and packet dataframes
    pos_df = pd.concat([meta_df_pos, pkt_df_pos], axis=1)
    neg_df = pd.concat([meta_df_neg, pkt_df_neg], axis=1)

    # Drop unwanted rows
    neg_df = neg_df.drop(neg_df[neg_df['Protocol'] == 17].index) # Dropped accidental UDP packets

    # Randomly sample from the biggest dataframe to match the size of the smallest dataframe
    pos_len = len(pos_df.index)
    neg_len = len(neg_df.index)
    if pos_len > neg_len:
        pos_df = pos_df.sample(n=neg_len, random_state=42)
    elif pos_len < neg_len:
        neg_df = neg_df.sample(n=pos_len, random_state=42)

    # Sort the DataFrame by 'Source IP', 'Destination IP', and 'Unix Timestamp'
    pos_df.sort_values(by=['Source IP', 'Destination IP', 'Unix Timestamp'], inplace=True)
    neg_df.sort_values(by=['Source IP', 'Destination IP', 'Unix Timestamp'], inplace=True)

    # Calculate the time elapsed for each group of rows
    pos_df['Time_Elapsed'] = pos_df.sort_values(by=['Unix Timestamp']).groupby(['Source IP', 'Destination IP'])['Unix Timestamp'].diff()
    neg_df['Time_Elapsed'] = neg_df.sort_values(by=['Unix Timestamp']).groupby(['Source IP', 'Destination IP'])['Unix Timestamp'].diff()

    # Replace NaN values in 'Time_Elapsed' with 0 (for the first row in each group)
    pos_df['Time_Elapsed'].fillna(0, inplace=True)
    neg_df['Time_Elapsed'].fillna(0, inplace=True)

    # Replace 6 with 0 and 17 with 1 in the specified column
    pos_df['Protocol'] = pos_df['Protocol'].replace({6: 0, 17: 1})
    neg_df['Protocol'] = neg_df['Protocol'].replace({6: 0, 17: 1})

    # Convert strings to lists
    pos_df[['Pkt_Direction', 'Pkt_Flags', 'Pkt_IATs', 'Pkt_Sizes']] = pos_df[['Pkt_Direction', 'Pkt_Flags', 'Pkt_IATs', 'Pkt_Sizes']].applymap(literal_eval)
    neg_df[['Pkt_Direction', 'Pkt_Flags', 'Pkt_IATs', 'Pkt_Sizes']] = neg_df[['Pkt_Direction', 'Pkt_Flags', 'Pkt_IATs', 'Pkt_Sizes']].applymap(literal_eval)

    # Drop unwanted columns
    pos_df = pos_df[['Time_Elapsed', 'Protocol', 'Pkt_Direction', 'Pkt_Flags', 'Pkt_IATs', 'Pkt_Sizes']]
    neg_df = neg_df[['Time_Elapsed', 'Protocol', 'Pkt_Direction', 'Pkt_Flags', 'Pkt_IATs', 'Pkt_Sizes']]

    # Apply the transformation function to relevant columns
    columns_to_transform = ['Pkt_Direction', 'Pkt_IATs', 'Pkt_Sizes', 'Pkt_Flags']
    for col in columns_to_transform:
        pos_df[col] = pos_df[col].apply(lambda x: transform_row_from_tuple(list(x), length) if isinstance(x, (list, tuple)) else transform_row_from_tuple([x], length))
    for col in columns_to_transform:
        neg_df[col] = neg_df[col].apply(lambda x: transform_row_from_tuple(list(x), length) if isinstance(x, (list, tuple)) else transform_row_from_tuple([x], length))

    # Custom function to flatten lists and lists of lists
    def flatten_element(cell):
        if isinstance(cell, list):
            return [item for sublist in cell for item in sublist] if any(isinstance(item, list) for item in cell) else cell
        return cell

    # Apply the custom function to all DataFrame columns
    pos_df = pos_df.applymap(flatten_element)
    neg_df = neg_df.applymap(flatten_element)

    # Save the dataframes
    pos_df.to_csv(f'{save_path}/pos.csv', index=True)
    neg_df.to_csv(f'{save_path}/neg.csv', index=True)

    # PRE-PROCESS FOR TENSORFLOW

    # Add the labels column
    pos_df['label'] = 0
    neg_df['label'] = 1

    # Concatenate the 2 dataframes
    data_df = pd.concat([pos_df, neg_df])

    # Convert 'Pkt_Direction,' 'Pkt_Flags,' 'Pkt_IATs,' and 'Pkt_Sizes' to lists
    data_df['Pkt_Direction'] = data_df['Pkt_Direction'].apply(list)
    data_df['Pkt_Flags'] = data_df['Pkt_Flags'].apply(list)
    data_df['Pkt_IATs'] = data_df['Pkt_IATs'].apply(list)
    data_df['Pkt_Sizes'] = data_df['Pkt_Sizes'].apply(list)

    # Ensure consistent data types (float32 for floating-point values, int32 for integers)
    # data_df['Time_Elapsed'] = data_df['Time_Elapsed'].astype('int32')                                             # This makes it negative, for some reason
    data_df['Protocol'] = data_df['Protocol'].astype('int32')
    data_df['label'] = data_df['label'].astype('int32')
        
    # Normalise columns
    data_df['Pkt_Sizes'] = data_df['Pkt_Sizes'].apply(scale_column, min=0, max=3000)                                # GLOBAL VARIABLES (Gotten from Training Data)w
    data_df['Pkt_IATs'] = data_df['Pkt_IATs'].apply(scale_column, min=0, max=5000)                                  # GLOBAL VARIABLES (Gotten from Training Data)
    data_df['Time_Elapsed'] = data_df['Time_Elapsed'].apply(scale_column, min=0, max=10000)                         # GLOBAL VARIABLES (Gotten from Training Data)
    data_df['Pkt_Flags'] = data_df['Pkt_Flags'].apply(scale_column, min=0, max=64)                                  # GLOBAL VARIABLES (Gotten from Training Data)

    # Save the dataframe
    data_df.to_csv(f'{save_path}/data.csv', index=True)

    return data_df




######################## Helper functions ########################

# Scale function
def scale_column(row, min, max):
    if isinstance(row, list):
        return [(r - min) / (max - min) if r < max else 1 for r in row ]
    else:
        return (row - min) / (max - min) if row < max else 1

# Function to convert a value to an 8-bit binary list
def value_to_binary_list(value):
    if isinstance(value, tuple):
        binary_lists = []
        for decimal_value in value:
            binary_string = format(decimal_value, '08b')  # Convert to 8-bit binary string
            binary_lists.append([int(bit) for bit in binary_string])  # Convert binary string to a list of integers
        return binary_lists
    elif isinstance(value, int):
        binary_string = format(value, '08b')  # Convert integer to 8-bit binary string
        return [int(bit) for bit in binary_string]  # Convert binary string to a list of integers
    else:
        return None  # Return None for unsupported types
    

# Define a custom function to calculate the length of a tuple or return 0 for non-list values
def get_tuple_length(lst):
    if isinstance(lst, tuple):
        return len(lst)
    else:
        return 0

# Define a custom function to calculate the length of a list or return 0 for non-list values
def get_list_length(lst):
    if isinstance(lst, list):
        return len(lst)
    else:
        return 0
    
# Function to transform a row with varying length to a row with fixed length (m)
def transform_row_from_tuple(row, m):
    if len(row) > m:
        # Truncate and calculate the average of the truncated values
        row = row[:m - 1] + [np.mean(row[m - 1:])]
    elif len(row) < m:
        # Fill with zeros until the desired length is reached
        row += [0] * (m - len(row))
    return row

# Function to pad and average nested lists
def transform_row_from_list_of_lists(row, m):
    if len(row) > m:
        # Truncate and calculate the average of the truncated values
        truncated = row[:m - 1]
        avg_value = np.mean(row[m - 1:], axis=0)
        return truncated + [avg_value.tolist()]
    elif len(row) < m:
        # Fill with zeros until the desired length is reached
        return row + [[0] * 8] * (m - len(row))
    return row

def get_adjusted_col_names(data_df):
    total_length = 0
    adjusted_col_names = []
    for col in data_df.columns:
        if isinstance(data_df[col].iloc[0], list):
            print(f"{col} is length {len(data_df[col].iloc[0])}")
            total_length += len(data_df[col].iloc[0])
            for i in range(0, len(data_df[col].iloc[0])):
                adjusted_col_names.append(f"{col}_{i}")
        else:
            print(f"{col} is length 1"),
            adjusted_col_names.append(col)
            total_length += 1
    return adjusted_col_names, total_length

def unravel_tf_dataset(dataset):
    dataset = list(dataset.as_numpy_iterator())

    batch_features = [batch_feature[0] for batch_feature in dataset]
    batch_features = [feature for feature in batch_features]
    features = [feature for sub_batch in batch_features for feature in sub_batch]
    

    batch_labels = [batch_label[1] for batch_label in dataset]
    batch_labels = [label for label in batch_labels]
    labels = [label for sub_batch in batch_labels for label in sub_batch]

    return features, labels

def get_cat_features(df, cat_feature_list=["Protocol", "Pkt_Direction", "Pkt_Flags"]):
    cat_feat_indices = []
    for idx, feat in enumerate(df.columns.tolist()):
        for cat_feat in cat_feature_list:
            if cat_feat in feat:
                cat_feat_indices.append(idx)
    return cat_feat_indices


