import h5py

def inspect_hdf5_file(file_path):
    with h5py.File(file_path, 'r') as file:
        # Iterate over each network configuration
        for net_key in file.keys():
            print(f"Network Configuration: {net_key}")
            net_group = file[net_key]

            # Iterate over each season within a network configuration
            for season_key in net_group.keys():
                print(f"  Season: {season_key}")
                season_group = net_group[season_key]

                # Iterate over each time step within a season
                for time_step_key in season_group.keys():
                    print(f"    Time Step: {time_step_key}")
                    time_step_group = season_group[time_step_key]

                    # Display datasets and their shapes within this time step
                    for dataset_name in time_step_group.keys():
                        dataset = time_step_group[dataset_name]
                        print(f"      Dataset: {dataset_name}, Shape: {dataset.shape}, Data Type: {dataset.dtype}")

def main():
    file_path = 'data/network_results.h5'  # Adjust the path as needed
    inspect_hdf5_file(file_path)  # Correctly calling the defined function

if __name__ == '__main__':
    main()

