import os
import time
import pickle
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import torch
from torch.utils.data import Dataset, DataLoader


# ========== prepare dataset ==========
def add_features(df):
    df = df.copy()

    # temporal features
    df['hour'] = df['date'].dt.hour
    df['month'] = df['date'].dt.month
    df['day_of_year'] = df['date'].dt.dayofyear

    # hourly cycle
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)

    # yearly seasonality
    df['year_sin'] = np.sin(2 * np.pi * df['day_of_year']/365)
    df['year_cos'] = np.cos(2 * np.pi * df['day_of_year']/365)

    # monthly seasonality
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)

    return df


def prepare_data(df, features, target='solar_irradiance', train_start='2017-01-01 00:00:00',
                 train_end='2022-12-31 23:59:59', valid_end='2023-12-31 23:59:59', test_end='2024-12-31 23:59:59'):
    # verify date order
    if not (pd.to_datetime(train_start) < pd.to_datetime(train_end) < pd.to_datetime(valid_end) < pd.to_datetime(test_end)):
        raise ValueError("Dates must be in order: train_start < train_end < valid_end < test_end")

    print(f"\nPreparing data with features: {features}")

    # generate target column
    df['target'] = df[target]

    # add features
    df = add_features(df)

    # separate spatial features from dynamic features
    spatial_features = ['latitude', 'longitude', 'height']
    dynamic_features = [f for f in features if f not in spatial_features]

    # global scaling for spatial features if they are in the features list
    spatial_features_in_input = [f for f in spatial_features if f in features]
    if spatial_features_in_input:
        spatial_scaler = StandardScaler()
        train_mask = (df['date'] >= train_start) & (df['date'] <= train_end)
        # get unique station spatial data to avoid duplicate scaling
        spatial_train_data = df[train_mask][['station'] + spatial_features_in_input].drop_duplicates('station')
        spatial_scaler.fit(spatial_train_data[spatial_features_in_input])

        # apply to all data
        for station in df['station'].unique():
            station_mask = df['station'] == station
            df.loc[station_mask, spatial_features_in_input] = spatial_scaler.transform(
                df.loc[station_mask, spatial_features_in_input].iloc[:1]
            ).repeat(station_mask.sum(), axis=0)

    # create scalers dictionary for each station's dynamic features
    scalers = {}
    for station in df['station'].unique():
        station_key = str(station)
        station_mask = df['station'] == station
        scalers[station_key] = StandardScaler()

        # fit scaler only on training data for each station
        train_mask = (df['date'] >= train_start) & (df['date'] <= train_end) & station_mask

        # columns to scale: dynamic features + target
        columns_to_scale = dynamic_features + [target] if target not in dynamic_features else dynamic_features

        # only fit and transform if we have columns to scale
        if columns_to_scale:
            scalers[station_key].fit(df.loc[train_mask, columns_to_scale])
            df.loc[station_mask, columns_to_scale] = scalers[station_key].transform(
                df.loc[station_mask, columns_to_scale]
            )

    # split data into train/valid/test sets
    train_data = df[(df['date'] >= train_start) & (df['date'] <= train_end)]
    valid_data = df[(df['date'] > train_end) & (df['date'] <= valid_end)]
    test_data = df[(df['date'] > valid_end) & (df['date'] <= test_end)]

    print("\nData split information:")
    print(f"Train period: {train_data['date'].min()} to {train_data['date'].max()}")
    print(f"Valid period: {valid_data['date'].min()} to {valid_data['date'].max()}")
    print(f"Test period: {test_data['date'].min()} to {test_data['date'].max()}")
    print(f"Sizes - train: {len(train_data)}, valid: {len(valid_data)}, test: {len(test_data)}")

    return train_data, valid_data, test_data, scalers


def create_sequences(data, feature_data, target_data, dates, stations, look_back, horizon):
    X, y = [], []
    date_seq, station_seq = [], []

    for i in range(len(data) - look_back - horizon + 1):
        X.append(feature_data[i:(i + look_back)])
        y.append(target_data[i + look_back:i + look_back + horizon])
        date_seq.append(dates[i + look_back])
        station_seq.append(stations[i + look_back])

    return np.array(X), np.array(y), date_seq, station_seq


class ASOSDataset(Dataset):
    def __init__(self, data, features, target, look_back, horizon, dataset_type='train'):
        self.features = features
        self.target = target
        self.look_back = look_back
        self.horizon = horizon
        self.dataset_type = dataset_type

        # create cache directory if it doesn't exist
        os.makedirs('cache', exist_ok=True)

        # create a unique cache filename based on dataset parameters
        stations = sorted(data['station'].unique())
        cache_filename = f"asos_dataset_{dataset_type}_lb{look_back}_h{horizon}_f{len(features)}_stations{len(stations)}.pkl"
        cache_path = os.path.join('cache', cache_filename)

        # try to load from cache first
        if os.path.exists(cache_path):
            print(f"loading ASOSDataset from cache: {cache_path}")
            start_time = time.time()
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
                self.X = cached_data['X']
                self.y = cached_data['y']
                self.dates = cached_data['dates']
                self.stations = cached_data['stations']
            print(f"loaded from cache in {time.time() - start_time:.2f} seconds")
        else:
            print(f"creating new ASOSDataset ({dataset_type})...")
            start_time = time.time()
            X_list, y_list = [], []
            date_list, station_list = [], []

            for station in data['station'].unique():
                station_data = data[data['station'] == station]

                # separate features and target
                feature_data = station_data[features].values
                target_data = station_data[target].values
                dates = station_data['date'].values
                stations = station_data['station'].values

                # create sequences
                X, y, dates_seq, stations_seq = create_sequences(
                    station_data, feature_data, target_data,
                    dates, stations, look_back, horizon
                )

                X_list.append(X)
                y_list.append(y)
                date_list.extend(dates_seq)
                station_list.extend(stations_seq)

            self.X = np.concatenate(X_list, axis=0)
            self.y = np.concatenate(y_list, axis=0)
            self.dates = np.array(date_list)
            self.stations = np.array(station_list)
            print(f"created in {time.time() - start_time:.2f} seconds")
            print(f"ASOSDataset features: {features}, length: {len(features)}")
            print(f"feature_data shape: {feature_data.shape}")

            # save to cache
            print(f"saving ASOSDataset to cache: {cache_path}")
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'X': self.X,
                    'y': self.y,
                    'dates': self.dates,
                    'stations': self.stations
                }, f)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = torch.FloatTensor(self.X[idx])
        y = torch.FloatTensor(self.y[idx])
        if self.horizon == 1:
            y = y.unsqueeze(-1)  # [1, 1]
        date = str(self.dates[idx])
        station = str(self.stations[idx])
        return X, y, date, station


class SpatialASOSDataset(Dataset):
    def __init__(self, data, features, target, look_back, horizon, dataset_type='train'):
        self.features = features
        self.target = target
        self.look_back = look_back
        self.horizon = horizon
        self.dataset_type = dataset_type

        # create cache directory if it doesn't exist
        os.makedirs('cache', exist_ok=True)

        # create a unique cache filename based on dataset parameters
        stations = sorted(data['station'].unique())
        cache_filename = f"spatial_asos_dataset_{dataset_type}_lb{look_back}_h{horizon}_f{len(features)}_stations{len(stations)}.pkl"
        cache_path = os.path.join('cache', cache_filename)

        # try to load from cache first
        if os.path.exists(cache_path):
            print(f"loading SpatialASOSDataset from cache: {cache_path}")
            start_time = time.time()
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
                self.X = cached_data['X']
                self.y = cached_data['y']
                self.spatial_info = cached_data['spatial_info']
                self.dates = cached_data['dates']
                self.stations = cached_data['stations']
            print(f"loaded from cache in {time.time() - start_time:.2f} seconds")
        else:
            print(f"creating new SpatialASOSDataset ({dataset_type})...")
            start_time = time.time()
            X_list, y_list = [], []
            spatial_list = []
            date_list, station_list = [], []

            for station in data['station'].unique():
                station_data = data[data['station'] == station]

                # separate features and target
                feature_data = station_data[features].values
                target_data = station_data[target].values
                dates = station_data['date'].values
                stations = station_data['station'].values

                # create sequences
                X, y, dates_seq, stations_seq = create_sequences(
                    station_data, feature_data, target_data,
                    dates, stations, look_back, horizon
                )

                X_list.append(X)
                y_list.append(y)
                date_list.extend(dates_seq)
                station_list.extend(stations_seq)

                # add spatial info
                spatial_info = torch.FloatTensor([
                    station_data['latitude'].iloc[0],
                    station_data['longitude'].iloc[0]
                ])
                spatial_list.extend([spatial_info] * len(X))

            self.X = np.concatenate(X_list, axis=0)
            self.y = np.concatenate(y_list, axis=0)
            self.spatial_info = torch.stack(spatial_list)
            self.dates = np.array(date_list)
            self.stations = np.array(station_list)
            print(f"created in {time.time() - start_time:.2f} seconds")
            print(f"SpatialASOSDataset features: {features}, length: {len(features)}")
            print(f"feature_data shape: {feature_data.shape}")

            # save to cache
            print(f"saving SpatialASOSDataset to cache: {cache_path}")
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'X': self.X,
                    'y': self.y,
                    'spatial_info': self.spatial_info,
                    'dates': self.dates,
                    'stations': self.stations
                }, f)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = torch.FloatTensor(self.X[idx])
        y = torch.FloatTensor(self.y[idx])
        if self.horizon == 1:
            y = y.unsqueeze(-1)
        return (
            X,
            self.spatial_info[idx],
            y,
            str(self.dates[idx]),
            str(self.stations[idx])
        )


class GraphASOSDataset(Dataset):
    def __init__(self, data, features, target, look_back, horizon, dataset_type='train', k_neighbors=2):
        self.features = features  # sequence features
        self.target = target
        self.look_back = look_back
        self.horizon = horizon
        self.k_neighbors = k_neighbors
        self.dataset_type = dataset_type

        # create cache directory if it doesn't exist
        os.makedirs('cache', exist_ok=True)

        # define edge feature columns
        self.edge_feature_columns = ['height']

        # create cache filename
        stations = sorted(data['station'].unique())
        self.stations = stations
        cache_filename = f"graph_dataset_{dataset_type}_k{k_neighbors}_lb{look_back}_h{horizon}_f{len(features)}_stations{len(stations)}.pkl"
        cache_path = os.path.join('cache', cache_filename)

        # try to load from cache first
        if os.path.exists(cache_path):
            print(f"loading graph dataset from cache: {cache_path}")
            start_time = time.time()
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
                self.edge_index = cached_data['edge_index']
                self.edge_weights = cached_data['edge_weights']
                self.edge_features = cached_data['edge_features']
                self.X = cached_data['X']
                self.y = cached_data['y']
                self.dates = cached_data['dates']
            print(f"loaded from cache in {time.time() - start_time:.2f} seconds")
        else:
            print(f"creating new graph dataset ({dataset_type})...")
            # get station info
            station_info = data[['station', 'latitude', 'longitude', 'height']].drop_duplicates()

            # create graph structure and sequences
            start_time = time.time()
            self._create_graph_structure(data, station_info)
            print(f"graph structure created in {time.time() - start_time:.2f} seconds")

            start_time = time.time()
            self.X, self.y, self.dates = self._prepare_sequences(data, features, target, look_back, horizon)
            print(f"sequences created in {time.time() - start_time:.2f} seconds")
            print(f"GraphASOSDataset features: {features}, length: {len(features)}")
            print(f"creating sequences with feature dim: {len(features)}")

            # save to cache
            print(f"saving graph dataset to cache: {cache_path}")
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'edge_index': self.edge_index,
                    'edge_weights': self.edge_weights,
                    'edge_features': self.edge_features,
                    'X': self.X,
                    'y': self.y,
                    'dates': self.dates
                }, f)

        print(f"initialized GraphASOSDataset ({dataset_type}) - {len(self.stations)} stations, "
              f"edge_index shape: {self.edge_index.shape}, "
              f"edge_features shape: {self.edge_features.shape}")

    # _create_graph_structure and _prepare_sequences methods remain the same
    def _create_graph_structure(self, data, station_info):
        # calculate distances between stations using sklearn's NearestNeighbors
        coords = station_info[['latitude', 'longitude']].values
        n_stations = len(station_info)

        # convert to radians for haversine distance
        coords_rad = np.radians(coords)

        # use a custom metric for haversine distance
        def haversine_distance(x, y):
            lat1, lon1 = x
            lat2, lon2 = y
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
            c = 2 * np.arcsin(np.sqrt(a))
            r = 6371  # earth radius in kilometers
            return c * r

        # find k+1 nearest neighbors (including self)
        nbrs = NearestNeighbors(n_neighbors=self.k_neighbors + 1, algorithm='ball_tree', metric=haversine_distance)
        nbrs.fit(coords_rad)
        distances, indices = nbrs.kneighbors(coords_rad)

        # initialize lists for graph construction
        edges = []
        edge_weights = []
        edge_features_list = []

        # mean distance for weight normalization
        mean_dist = np.mean(distances[:, 1:])

        # k-nearest neighbors graph construction
        for i in range(n_stations):
            # skip the first neighbor (which is self)
            for j_idx in range(1, self.k_neighbors + 1):
                j = indices[i, j_idx]
                dist = distances[i, j_idx]

                # calculate distance-based weight (vectorized)
                dist_weight = np.exp(-dist ** 2 / (2 * (mean_dist / 3) ** 2))

                # calculate feature-based differences
                feature_diffs = []
                for feat in self.edge_feature_columns:
                    station_i_data = data[data['station'] == self.stations[i]][feat].mean()
                    station_j_data = data[data['station'] == self.stations[j]][feat].mean()

                    # normalize by feature's standard deviation
                    feat_std = data[feat].std()
                    diff = abs(station_i_data - station_j_data) / feat_std
                    feature_diffs.append(diff)

                # combine distance and feature differences
                feature_weight = np.exp(-np.mean(feature_diffs))
                final_weight = dist_weight * feature_weight

                # add bidirectional edges
                edges.extend([[i, j], [j, i]])
                edge_weights.extend([final_weight, final_weight])
                edge_features_list.extend([feature_diffs, feature_diffs])

        # store graph representations
        self.edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        self.edge_weights = torch.tensor(edge_weights, dtype=torch.float)
        self.edge_features = torch.tensor(edge_features_list, dtype=torch.float)

        print(f"created graph structure with {len(edges)} edges")

    def _prepare_sequences(self, data, features, target, look_back, horizon):
        dates = sorted(data['date'].unique())
        num_samples = len(dates) - look_back - horizon + 1
        num_stations = len(self.stations)

        # create a lookup dictionary for faster access
        data_lookup = {}
        for station in self.stations:
            station_data = data[data['station'] == station]
            data_lookup[station] = {
                str(date): row for date, row in
                zip(station_data['date'], station_data[features + [target]].values)
            }

        # initialize arrays
        X = np.zeros((num_samples, look_back, num_stations, len(features)))
        if horizon == 1:
            y = np.zeros((num_samples, num_stations, 1))  # [samples, stations, 1]
        else:
            y = np.zeros((num_samples, num_stations, horizon))  # [samples, stations, horizon]
        sequence_dates = []

        # create sequences with optimized lookups
        for i in range(num_samples):
            # input sequence
            for t in range(look_back):
                date = dates[i + t]
                date_str = str(date)
                for s_idx, station in enumerate(self.stations):
                    if date_str in data_lookup[station]:
                        X[i, t, s_idx] = data_lookup[station][date_str][:-1]  # exclude target

            # target sequence
            if horizon == 1:
                target_date = dates[i + look_back]
                target_date_str = str(target_date)
                for s_idx, station in enumerate(self.stations):
                    if target_date_str in data_lookup[station]:
                        y[i, s_idx, 0] = data_lookup[station][target_date_str][-1]  # target only
            else:
                for h in range(horizon):
                    target_date = dates[i + look_back + h]
                    target_date_str = str(target_date)
                    for s_idx, station in enumerate(self.stations):
                        if target_date_str in data_lookup[station]:
                            y[i, s_idx, h] = data_lookup[station][target_date_str][-1]

            sequence_dates.append(dates[i + look_back])

        print(f"created sequences - X shape: {X.shape}, y shape: {y.shape}")
        return X, y, np.array(sequence_dates)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = torch.FloatTensor(self.X[idx])
        y = torch.FloatTensor(self.y[idx])
        if self.horizon == 1:
            y = y.unsqueeze(-1)
        return (
            X,
            self.edge_index,
            self.edge_weights,
            self.edge_features,
            y,
            str(self.dates[idx]),
            [str(s) for s in self.stations]
        )

    def get_edge_index(self):
        return self.edge_index, self.edge_weights, self.edge_features


def graph_collate_fn(batch):
    x = torch.stack([item[0] for item in batch])  # [batch, seq, nodes, features]
    edge_index = batch[0][1]  # edge_index
    edge_weights = batch[0][2]  # edge_weights
    edge_features = batch[0][3]  # edge_features
    y = torch.stack([item[4] for item in batch])  # [batch, nodes]
    dates = [item[5] for item in batch]  # list of dates
    stations = [item[6] for item in batch]  # list of stations

    return x, edge_index, edge_weights, edge_features, y, dates, stations


# ========== prepare dataloaders ==========
def get_dataloaders(train_data, valid_data, test_data, features, target, look_back, horizon, batch_size):
    print("\nDataloader creation check:")
    print(f"Test data date range in dataloader: {test_data['date'].min()} to {test_data['date'].max()}")

    datasets = {}

    # basic ASOS datasets (for LSTM, CNN, Transformer)
    datasets['basic'] = {
        'train': ASOSDataset(train_data, features, target, look_back, horizon, dataset_type='train'),
        'valid': ASOSDataset(valid_data, features, target, look_back, horizon, dataset_type='valid'),
        'test': ASOSDataset(test_data, features, target, look_back, horizon, dataset_type='test')
    }

    # spatial ASOS datasets (for STTransformer)
    datasets['spatial'] = {
        'train': SpatialASOSDataset(train_data, features, target, look_back, horizon, dataset_type='train'),
        'valid': SpatialASOSDataset(valid_data, features, target, look_back, horizon, dataset_type='valid'),
        'test': SpatialASOSDataset(test_data, features, target, look_back, horizon, dataset_type='test')
    }

    # graph ASOS datasets (for GCN-based models)
    datasets['graph'] = {
        'train': GraphASOSDataset(train_data, features, target, look_back, horizon, dataset_type='train'),
        'valid': GraphASOSDataset(valid_data, features, target, look_back, horizon, dataset_type='valid'),
        'test': GraphASOSDataset(test_data, features, target, look_back, horizon, dataset_type='test')
    }

    # create dataloaders
    dataloaders = {}

    # basic dataloaders
    dataloaders['basic'] = {
        'train': DataLoader(datasets['basic']['train'], batch_size=batch_size, shuffle=True),
        'valid': DataLoader(datasets['basic']['valid'], batch_size=batch_size),
        'test': DataLoader(datasets['basic']['test'], batch_size=batch_size)
    }

    # spatial dataloaders
    dataloaders['spatial'] = {
        'train': DataLoader(datasets['spatial']['train'], batch_size=batch_size, shuffle=True),
        'valid': DataLoader(datasets['spatial']['valid'], batch_size=batch_size),
        'test': DataLoader(datasets['spatial']['test'], batch_size=batch_size)
    }

    # graph dataloaders with special collate function
    dataloaders['graph'] = {
        'train': DataLoader(datasets['graph']['train'], batch_size=batch_size, shuffle=True, collate_fn=graph_collate_fn),
        'valid': DataLoader(datasets['graph']['valid'], batch_size=batch_size, collate_fn=graph_collate_fn),
        'test': DataLoader(datasets['graph']['test'], batch_size=batch_size, collate_fn=graph_collate_fn)
    }

    # get edge_index from graph dataset for GCN-based models
    edge_index = datasets['graph']['train'].get_edge_index()

    # add check for test dataloader
    for loader_type in ['basic', 'spatial', 'graph']:
        first_batch = next(iter(dataloaders[loader_type]['test']))
        if loader_type == 'basic':
            first_dates = first_batch[2][:5]
        elif loader_type == 'spatial':
            first_dates = first_batch[3][:5]
        else:  # graph
            first_dates = first_batch[5][:5]
        print(f"{loader_type} test loader first batch dates: {first_dates}")

    return dataloaders, edge_index