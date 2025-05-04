import numpy as np
import pandas as pd
import xarray as xr
import cdsapi
from datetime import datetime, timedelta
from mpi4py import MPI
import os

def get_years_and_months(start_date, end_date):
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    years = set()
    months = set()
    
    current = start.replace(day=1)
    
    while current <= end:
        years.add(str(current.year))
        months.add(f"{current.month:02d}")
 
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)
    
    return sorted(years), sorted(months)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()  # Process ID
size = comm.Get_size()  # Number of processes

# Ensure 'data/' directory exists
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def get_monthly_chunks(start_date, end_date):
    """Splits the date range into 1-month chunks."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    chunks = []
    current_start = start.replace(day=1)

    while current_start <= end:
        next_end = (current_start + timedelta(days=32)).replace(day=1) - timedelta(days=1)  # Last day of the month
        if next_end > end:
            next_end = end
        
        chunks.append((current_start.strftime("%Y-%m-%d"), next_end.strftime("%Y-%m-%d")))
        current_start = next_end + timedelta(days=1)
    
    return chunks

def download_era5_data(dataset, start, end, area, variables, pressure_levels=None):
    """Downloads ERA5 data only if it doesn't already exist."""
    name = os.path.join(DATA_DIR, f"{dataset}-{start}-{end}.nc")

    # Check if file already exists
    if os.path.exists(name) and os.path.getsize(name) > 0:
        print(f"Skipping already downloaded file: {name}")
        return name

    year = start[:4]
    month = start[5:7]

    request = {
        "product_type": dataset,
        "variable": variables,
        "year": [year],
        "month": [month],
        "day": [f"{day:02d}" for day in range(1, 32)],
        "time": [f"{hour:02d}:00" for hour in range(24)],
        "data_format": "netcdf",
        "area": area
    }

    if pressure_levels:
        request["pressure_level"] = pressure_levels

    client = cdsapi.Client()
    client.retrieve(dataset, request, name)
    
    print(f"Downloaded: {name}")
    return name

def merge_netcdf_files(files, output_name):
    """Merges multiple NetCDF files into one dataset."""
    output_path = os.path.join(DATA_DIR, output_name)

    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        print(f"Merged file already exists: {output_path}")
        return output_path

    datasets = [xr.open_dataset(f) for f in files]
    merged_ds = xr.concat(datasets, dim="time")
    merged_ds.to_netcdf(output_path)

    print(f"Merged NetCDF saved as: {output_path}")
    return output_path

def parallel_download_era5(dataset, start, end, area, variables, pressure_levels=None):
    """Uses MPI to download ERA5 data in parallel by distributing monthly chunks across processes."""
    chunks = get_monthly_chunks(start, end)
    files = []

    # Distribute work among MPI processes
    for i, (chunk_start, chunk_end) in enumerate(chunks):
        if i % size == rank:  # Each process gets its own months to download
            file = download_era5_data(dataset, chunk_start, chunk_end, area, variables, pressure_levels)
            files.append(file)

    # Gather all downloaded files at root process
    all_files = comm.gather(files, root=0)

    # Root process merges all files
    if rank == 0:
        merged_files = [f for sublist in all_files for f in sublist]  # Flatten list
        return merge_netcdf_files(merged_files, f"{dataset}-{start}-{end}.nc")
    
    return None

def pivot_era5_single(name):
    ds = xr.open_dataset(name)
    df = ds.to_dataframe()
    
    df = df.reset_index(level='time', drop=True)
    df.dropna(inplace=True)
    
    df.drop(columns=[col for col in ['number', 'expver'] if col in df.columns], inplace=True)
    df_reset = df.reset_index()

    # Filter to keep only integer lat/lon values
    mask = (
        (df_reset['latitude'] % 1 == 0) &
        (df_reset['longitude'] % 1 == 0)
    )
    df_reset = df_reset[mask]

    feature_columns = df_reset.columns[3:]
    df_pivot = df_reset.pivot(index="valid_time", columns=["latitude", "longitude"], values=feature_columns)
    df_pivot.columns = [f"{lat}-{lon}_{feature}" for feature, lat, lon in df_pivot.columns]

    return df_pivot

def pivot_era5_pressure(name):
    ds = xr.open_dataset(name)
    df = ds.to_dataframe()
    df = df.reset_index(level='time', drop=True)
    df.dropna(inplace=True)

    df.drop(columns=[col for col in ['number', 'expver'] if col in df.columns], inplace=True)
    df_reset = df.reset_index()

    # Filter to keep only integer lat/lon values
    mask = (
        (df_reset['latitude'] % 1 == 0) &
        (df_reset['longitude'] % 1 == 0)
    )
    df_reset = df_reset[mask]

    feature_columns = df_reset.columns[4:]
    df_pivot = df_reset.pivot(index="valid_time", columns=["latitude", "longitude", "pressure_level"], values=feature_columns)
    df_pivot.columns = [f"{lat}-{lon}_{pressure}_{feature}" for feature, lat, lon, pressure in df_pivot.columns]

    return df_pivot

def merge_era5(name_s, name_p):
    df_s = pivot_era5_single(name_s)
    df_p = pivot_era5_pressure(name_p)

    df = df_s.merge(df_p, on='valid_time', how='outer')
    df['valid_time'] = df.index
    df = df.reset_index(drop=True) 

    return df

if __name__ == "__main__":
    start = '2024-01-02'
    end = '2025-01-01'
    area =  [41, -110, 37, -102]

    name_s = parallel_download_era5(
        "reanalysis-era5-single-levels", start, end, area,
        ["10m_u_component_of_wind", "10m_v_component_of_wind", "2m_temperature", "mean_sea_level_pressure"]
    )

    name_p = parallel_download_era5(
        "reanalysis-era5-pressure-levels", start, end, area,
        ["geopotential", "u_component_of_wind", "v_component_of_wind"],
        pressure_levels=["100", "250", "450", "650", "800", "900", "1000"]
    )

    if rank == 0:
        df_era5 = merge_era5(name_s, name_p)
        df_era5['valid_time'] = pd.to_datetime(df_era5['valid_time']).dt.floor('H')

        df_bdu = pd.read_csv('bdu_clean.csv')
        df_bdu['date'] = pd.to_datetime(df_bdu['date']).dt.floor('H')

        df_bdu.drop(columns=['month_name', 'year'], inplace=True)

        df = df_bdu.merge(df_era5, left_on='date', right_on='valid_time', how='inner')
        df.to_csv(os.path.join(DATA_DIR, 'clean_df.csv'), index=False)

        print(f"Merged dataset saved in {DATA_DIR}/clean_df.csv")

