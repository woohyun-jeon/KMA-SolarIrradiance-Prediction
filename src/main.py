import os
from datetime import datetime
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import torch
import torch.nn as nn

from utils import load_config, set_seed, EarlyStopping
from datasets import prepare_data, get_dataloaders
from models import get_model


def handle_missing_values(df):
    for station in df['station'].unique():
        mask = df['station'] == station
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        station_data = df.loc[mask, numeric_columns]

        for col in numeric_columns:
            if col == 'solar_irradiance':
                station_data[col] = station_data[col].fillna(0)
            else:
                col_mean = station_data[col].mean()
                station_data[col] = station_data[col].fillna(col_mean)
                station_data[col] = station_data[col].interpolate(method='linear')
                station_data[col] = station_data[col].fillna(method='ffill').fillna(method='bfill')

        df.loc[mask, numeric_columns] = station_data

    return df


def train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, device,
                num_epochs, patience=10, model_type=None, gradient_clip=1.0, model_dir=None):
    if model_dir is None:
        model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)

    model_filename = os.path.join(model_dir, f'best_model_{model_type.lower()}.pth')
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()

            try:
                if model_type in ['GCNLSTM', 'GCNCNN', 'GCNTransformer', 'STGCNLSTM']:
                    x, edge_index, edge_weights, edge_features, y, dates, stations = batch
                    x = x.to(device)
                    edge_index = edge_index.to(device)
                    edge_weights = edge_weights.to(device)
                    edge_features = edge_features.to(device)
                    y = y.to(device)
                    outputs = model(x, edge_index, edge_weights, edge_features)

                elif model_type == 'STTransformer':
                    x, spatial_info, y, dates, stations = batch
                    x, spatial_info, y = x.to(device), spatial_info.to(device), y.to(device)
                    outputs = model(x, spatial_info)

                else:  # LSTM, CNN, Transformer
                    x, y, dates, stations = batch
                    x, y = x.to(device), y.to(device)
                    outputs = model(x)

                loss = criterion(outputs, y)
                loss.backward()

                # gradient clipping
                if gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

                optimizer.step()
                train_loss += loss.item()

            except Exception as e:
                print(f"error in batch {batch_idx}: {str(e)}")
                print(f"batch contents: {[b.shape if torch.is_tensor(b) else type(b) for b in batch]}")
                raise e

        train_loss /= len(train_loader)

        # validation phase
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for batch in valid_loader:
                try:
                    if model_type in ['GCNLSTM', 'GCNCNN', 'GCNTransformer', 'STGCNLSTM']:
                        x, edge_index, edge_weights, edge_features, y, dates, stations = batch
                        x = x.to(device)
                        edge_index = edge_index.to(device)
                        edge_weights = edge_weights.to(device)
                        edge_features = edge_features.to(device)
                        y = y.to(device)
                        outputs = model(x, edge_index, edge_weights, edge_features)

                    elif model_type == 'STTransformer':
                        x, spatial_info, y, dates, stations = batch
                        x, spatial_info, y = x.to(device), spatial_info.to(device), y.to(device)
                        outputs = model(x, spatial_info)

                    else:  # LSTM, CNN, Transformer
                        x, y, dates, stations = batch
                        x, y = x.to(device), y.to(device)
                        outputs = model(x)

                    valid_loss += criterion(outputs, y).item()

                except Exception as e:
                    print(f"error in validation: {str(e)}")
                    print(f"batch contents: {[b.shape if torch.is_tensor(b) else type(b) for b in batch]}")
                    raise e

        valid_loss /= len(valid_loader)

        # learning rate scheduling
        scheduler.step(valid_loss)

        print(f'epoch {epoch + 1}: train loss = {train_loss:.4f}, valid loss = {valid_loss:.4f}')

        # early stopping
        early_stopping(valid_loss, model, model_filename)
        if early_stopping.early_stop:
            print(f'early stopping triggered after {epoch + 1} epochs')
            break

    return early_stopping.best_loss


def evaluate_model(model, test_loader, criterion, device, model_type=None, scalers=None, save_dir=None):
    model.eval()
    station_predictions = {}
    station_actuals = {}
    station_dates = {}
    test_loss = 0

    with torch.no_grad():
        for batch in test_loader:
            try:
                if model_type in ['GCNLSTM', 'GCNCNN', 'GCNTransformer', 'STGCNLSTM']:
                    x, edge_index, edge_weights, edge_features, y, dates, stations = batch
                    x = x.to(device)
                    edge_index = edge_index.to(device)
                    edge_weights = edge_weights.to(device)
                    edge_features = edge_features.to(device)
                    y = y.to(device)

                    outputs = model(x, edge_index, edge_weights, edge_features)  # [batch, nodes, horizon]

                    # collect predictions for each station
                    for b in range(x.size(0)):
                        for s in range(x.size(2)):
                            station_id = str(stations[b][s])
                            date = pd.to_datetime(dates[b])

                            # initialize if not exists
                            if station_id not in station_predictions:
                                station_predictions[station_id] = {}
                                station_actuals[station_id] = {}
                                station_dates[station_id] = []

                            # save predictions and actuals as dictionary with date as key
                            if date not in station_predictions[station_id]:
                                station_predictions[station_id][date] = []
                                station_actuals[station_id][date] = []
                                station_dates[station_id].append(date)

                            # collect all predictions for this timestamp
                            station_predictions[station_id][date].extend(
                                [outputs[b, s, h].cpu().item() for h in range(outputs.size(-1))])
                            station_actuals[station_id][date].extend(
                                [y[b, s, h].cpu().item() for h in range(y.size(-1))])

                elif model_type == 'STTransformer':
                    x, spatial_info, y, dates, stations = batch
                    x, spatial_info, y = x.to(device), spatial_info.to(device), y.to(device)
                    outputs = model(x, spatial_info)

                    for i, station in enumerate(stations):
                        station_id = str(station)
                        date = pd.to_datetime(dates[i])

                        if station_id not in station_predictions:
                            station_predictions[station_id] = {}
                            station_actuals[station_id] = {}
                            station_dates[station_id] = []

                        if date not in station_predictions[station_id]:
                            station_predictions[station_id][date] = []
                            station_actuals[station_id][date] = []
                            station_dates[station_id].append(date)

                        station_predictions[station_id][date].extend(
                            [outputs[i, h].cpu().item() for h in range(outputs.size(-1))])
                        station_actuals[station_id][date].extend([y[i, h].cpu().item() for h in range(y.size(-1))])

                else:  # LSTM, CNN, Transformer
                    x, y, dates, stations = batch
                    x, y = x.to(device), y.to(device)
                    outputs = model(x)

                    for i, station in enumerate(stations):
                        station_id = str(station)
                        date = pd.to_datetime(dates[i])

                        if station_id not in station_predictions:
                            station_predictions[station_id] = {}
                            station_actuals[station_id] = {}
                            station_dates[station_id] = []

                        if date not in station_predictions[station_id]:
                            station_predictions[station_id][date] = []
                            station_actuals[station_id][date] = []
                            station_dates[station_id].append(date)

                        station_predictions[station_id][date].extend(
                            [outputs[i, h].cpu().item() for h in range(outputs.size(-1))])
                        station_actuals[station_id][date].extend([y[i, h].cpu().item() for h in range(y.size(-1))])

                loss = criterion(outputs, y)
                test_loss += loss.item()

            except Exception as e:
                print(f"error in evaluation: {str(e)}")
                print(f"batch contents: {[b.shape if torch.is_tensor(b) else type(b) for b in batch]}")
                raise e

    test_loss /= len(test_loader)
    print(f"test loss: {test_loss:.4f}")

    try:
        # calculate metrics for each station
        station_metrics = {}
        all_predictions = []
        all_actuals = []

        for station_id in station_predictions.keys():
            # convert dictionary to dataframe with averaged predictions
            dates = sorted(station_dates[station_id])  # sort dates to ensure correct order
            preds = []
            acts = []

            for date in dates:
                # average predictions for each timestamp
                pred_avg = np.mean(station_predictions[station_id][date])
                # use first actual value (they should all be the same for the same timestamp)
                act_val = station_actuals[station_id][date][0]

                preds.append(pred_avg)
                acts.append(act_val)

            preds = np.array(preds)
            acts = np.array(acts)

            # inverse transform if scalers provided
            if scalers is not None and station_id in scalers:
                scaler = scalers[station_id]

                # create dummy arrays for inverse transform
                preds_2d = preds.reshape(-1, 1)
                acts_2d = acts.reshape(-1, 1)

                # get the target index in the scaler features
                target_idx = scaler.n_features_in_ - 1  # assuming target is the last feature

                # prepare dummy arrays with zeros except for our values
                dummy_preds = np.zeros((len(preds), scaler.n_features_in_))
                dummy_acts = np.zeros((len(acts), scaler.n_features_in_))

                # place values at the correct index
                dummy_preds[:, target_idx] = preds_2d.ravel()
                dummy_acts[:, target_idx] = acts_2d.ravel()

                # inverse transform
                preds = scaler.inverse_transform(dummy_preds)[:, target_idx]
                acts = scaler.inverse_transform(dummy_acts)[:, target_idx]

                # ensure night values or missing values are exactly 0
                zero_mask = acts < 0.0005  # threshold close to zero
                acts[zero_mask] = 0.0

            # calculate metrics
            mae = mean_absolute_error(acts, preds)
            rmse = np.sqrt(mean_squared_error(acts, preds))
            r2 = r2_score(acts, preds)

            station_metrics[station_id] = {
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'n_samples': len(preds)
            }

            # save station results
            station_df = pd.DataFrame({
                'date': dates,
                'station': station_id,
                'predicted': preds,
                'actual': acts
            })
            save_path = os.path.join(
                save_dir, f'test_results_{model_type.lower()}_station_{station_id}.csv'
            ) if save_dir else f'results/test_results_{model_type.lower()}_station_{station_id}.csv'
            station_df.to_csv(save_path, index=False, float_format='%.6f')

            all_predictions.extend(preds)
            all_actuals.extend(acts)

            print(
                f"station {station_id} metrics - mae: {mae:.4f}, rmse: {rmse:.4f}, r2: {r2:.4f}, n_samples: {len(preds)}"
            )

        # calculate overall metrics
        overall_mae = mean_absolute_error(all_actuals, all_predictions)
        overall_rmse = np.sqrt(mean_squared_error(all_actuals, all_predictions))
        overall_r2 = r2_score(all_actuals, all_predictions)

        print(f"overall metrics - mae: {overall_mae:.4f}, rmse: {overall_rmse:.4f}, r2: {overall_r2:.4f}")

        return {
            'station_metrics': station_metrics,
            'overall_metrics': {
                'mae': overall_mae,
                'rmse': overall_rmse,
                'r2': overall_r2
            },
            'test_loss': test_loss
        }

    except Exception as e:
        print(f"error in metric calculation: {str(e)}")
        print(f"predictions shape: {[len(v) for v in station_predictions.values()]}")
        print(f"actuals shape: {[len(v) for v in station_actuals.values()]}")
        raise e


def main():
    try:
        # check and load configuration
        if not os.path.exists('configs.yaml'):
            raise FileNotFoundError("configs.yaml file not found")

        config = load_config()

        # validate required paths in config
        required_paths = ['data_path', 'save_dir', 'model_dir']
        for path in required_paths:
            if path not in config['path']:
                raise KeyError(f"'{path}' not found in config file")

        # check if data file exists
        if not os.path.exists(config['path']['data_path']):
            raise FileNotFoundError(f"data file not found at {config['path']['data_path']}")

        # create all necessary directories
        os.makedirs(config['path']['save_dir'], exist_ok=True)
        os.makedirs(config['path']['model_dir'], exist_ok=True)
        os.makedirs('cache', exist_ok=True)  # for dataset caching

        # set configuration
        set_seed(seed=config['params']['seed'])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"using device: {device}")

        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

        # load data
        df = pd.read_csv(config['path']['data_path'])
        df['date'] = pd.to_datetime(df['date'])
        df = handle_missing_values(df)
        num_nodes = len(df['station'].unique())
        print(f"data shape: {df.shape}")
        print(f"number of nodes: {num_nodes}")

        # define base features without spatial information
        base_features = [
            'wind_speed', 'wind_direction', 'relative_humidity', 'local_pressure',
            'sea_level_pressure', 'cloud_cover', 'visibility', 'sunshine_duration',
            'solar_irradiance',
            'hour_sin', 'hour_cos', 'year_sin', 'year_cos', 'month_sin', 'month_cos'
        ]
        spatial_features = ['latitude', 'longitude', 'height']

        # prepare all dataloaders first
        dataloaders = {}

        # LSTM, CNN, Transformer - use full features
        full_features = base_features + spatial_features
        train_data_full, valid_data_full, test_data_full, scalers_full = prepare_data(
            df,
            features=full_features,
            train_start=config['data']['train_start'],
            train_end=config['data']['train_end'],
            valid_end=config['data']['valid_end'],
            test_end=config['data']['test_end']
        )
        basic_loaders, _ = get_dataloaders(
            train_data_full, valid_data_full, test_data_full,
            features=full_features,
            target='target',
            look_back=config['params']['look_back'],
            horizon=config['params']['horizon'],
            batch_size=config['params']['batch_size']
        )
        dataloaders['basic'] = basic_loaders['basic']

        # STTransformer, STGCNLSTM - use base features
        train_data_base, valid_data_base, test_data_base, scalers_base = prepare_data(
            df,
            features=base_features,
            train_start=config['data']['train_start'],
            train_end=config['data']['train_end'],
            valid_end=config['data']['valid_end'],
            test_end=config['data']['test_end']
        )

        spatial_loaders, _ = get_dataloaders(
            train_data_base, valid_data_base, test_data_base,
            features=base_features,
            target='target',
            look_back=config['params']['look_back'],
            horizon=config['params']['horizon'],
            batch_size=config['params']['batch_size']
        )
        dataloaders['spatial'] = spatial_loaders['spatial']

        graph_loaders, graph_data = get_dataloaders(
            train_data_base, valid_data_base, test_data_base,
            features=base_features,
            target='target',
            look_back=config['params']['look_back'],
            horizon=config['params']['horizon'],
            batch_size=config['params']['batch_size']
        )
        dataloaders['graph'] = graph_loaders['graph']

        # move graph data to device if available
        if graph_data is not None:
            edge_index, edge_weights, edge_features = graph_data
            edge_index = edge_index.to(device)
            edge_weights = edge_weights.to(device)
            edge_features = edge_features.to(device)

        # define model configurations
        model_configs = {
            'LSTM': {
                'features': full_features,
                'input_dim': len(full_features),
                'dataloader_type': 'basic',
                'scalers': scalers_full
            },
            'CNN': {
                'features': full_features,
                'input_dim': len(full_features),
                'dataloader_type': 'basic',
                'scalers': scalers_full
            },
            'Transformer': {
                'features': full_features,
                'input_dim': len(full_features),
                'dataloader_type': 'basic',
                'scalers': scalers_full
            },
            'STTransformer': {
                'features': base_features,
                'input_dim': len(base_features),
                'dataloader_type': 'spatial',
                'scalers': scalers_base
            },
            'STGCNLSTM': {
                'features': base_features,
                'input_dim': len(base_features),
                'dataloader_type': 'graph',
                'scalers': scalers_base
            }
        }

        results = {}

        # train and evaluate each model
        for model_type, config_dict in model_configs.items():
            print(f"\ntraining {model_type}...")

            try:
                # get appropriate dataloaders
                loader_type = config_dict['dataloader_type']
                train_loader = dataloaders[loader_type]['train']
                valid_loader = dataloaders[loader_type]['valid']
                test_loader = dataloaders[loader_type]['test']

                # create model
                model = get_model(
                    model_type=model_type,
                    input_dim=config_dict['input_dim'],
                    hidden_dim=config['params']['hidden_dims'],
                    output_dim=config['params']['horizon'],
                    num_layers=config['params']['num_layers'],
                    num_nodes=num_nodes,
                    embed_dim=config['params']['embed_dims'],
                    seq_length=config['params']['look_back'],
                    dropout=config['params']['dropout']
                ).to(device)

                # set up training
                criterion = nn.MSELoss()
                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=config['params']['learning_rate'],
                    weight_decay=config['params']['weight_decay']
                )
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
                )

                # train model
                best_val_loss = train_model(
                    model, train_loader, valid_loader, criterion, optimizer,
                    scheduler, device, config['params']['num_epochs'],
                    patience=config['params']['patience'],
                    model_type=model_type,
                    gradient_clip=config['params']['gradient_clip'],
                    model_dir=config['path']['model_dir']
                )

                # load best model for evaluation
                model_path = os.path.join(config['path']['model_dir'], f'best_model_{model_type.lower()}.pth')
                model.load_state_dict(torch.load(model_path))

                # evaluate model
                eval_results = evaluate_model(
                    model, test_loader, criterion, device,
                    model_type=model_type, scalers=config_dict['scalers'],
                    save_dir=config['path']['save_dir']
                )

                results[model_type] = {
                    'eval_results': eval_results,
                    'best_val_loss': float(best_val_loss),
                    'features_used': config_dict['features']
                }

                # save individual model results
                save_path = os.path.join(
                    config['path']['save_dir'],
                    f"results_{model_type.lower()}_{current_time}.json"
                )
                with open(save_path, "w") as f:
                    json.dump(results[model_type], f, indent=4)

            except Exception as e:
                print(f"error in {model_type}: {str(e)}")
                continue

        # save all results
        save_path = os.path.join(
            config['path']['save_dir'],
            f"results_all_{current_time}.json"
        )
        with open(save_path, "w") as f:
            json.dump(results, f, indent=4)

        print(f"\nresults saved to {save_path}")

    except Exception as e:
        print(f"error in initialization: {str(e)}")
        raise


if __name__ == "__main__":
    main()