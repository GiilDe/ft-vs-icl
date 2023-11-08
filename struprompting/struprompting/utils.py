import json
import os

class LazyWriter:
    writers = {}  # Class-level dictionary to store LazyWriter instances

    def __init__(self, name, keys, batch_size):
        self.name = name
        self.filenames = [f'{name}/{key}.npz' for key in keys]
        self.batch_size = batch_size
        self.data = {key: [None] * batch_size for key in keys}
        self.current_index = 0  # Keep track of the current position

        # Register the writer instance
        LazyWriter.writers[name] = self

    @classmethod
    def get(cls, name):
        if name in cls.writers:
            return cls.writers[name]

    def append(self, data_dict):
        if not isinstance(data_dict, dict):
            raise ValueError("Input data_dict must be a dictionary.")

        if self.current_index >= self.batch_size:
            raise ValueError("The batch is full. You need to save the current data batch before appending more data.")

        # Append the data to the current position in the batch
        for key in data_dict:
            self.data[key][self.current_index] = data_dict[key]
        self.current_index += 1

    def save(self):
        # Remove the actual JSONL file from the operating system if it exists
        for filename in self.filenames:
            if os.path.exists(filename):
                os.remove(filename)
            with open(filename, 'w') as f:
                for key, item in self.data.items():
                    np.savez(f, *item)

        # Delete the data from the memory
        del LazyWriter.writers[self.name]
        del self
