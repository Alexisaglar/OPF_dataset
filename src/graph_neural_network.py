import pickle

# List to hold the loaded network configurations
loaded_nets = []

# Open the file in binary read mode
with open('data/successful_nets.pkl', 'rb') as f:
    loaded_nets = pickle.load(f)

for i, _ in enumerate(loaded_nets):
    print(loaded_nets[i].line['in_service'])

print(f"Loaded {len(loaded_nets)} successful configurations.")

