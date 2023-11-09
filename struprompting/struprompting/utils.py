import jsonlines
import os
import numpy as np
from multiprocessing import Value

class LazyWriter:
    writers = {}  # Class-level dictionary to store LazyWriter instances

    def __init__(self, name, keys, max_size):
        self.name = name
        self.max_size = max_size
        self.data = {key: [None] * max_size for key in keys}
        self.counter = Value('i', -1)  # Keep track of the current position

        # Register the writer instance
        LazyWriter.writers[name] = self

    @classmethod
    def get(cls, name):
        if name in cls.writers:
            return cls.writers[name]

    def append(self, data_dict):
        if not isinstance(data_dict, dict):
            raise ValueError("Input data_dict must be a dictionary.")

        if self.counter >= self.max_size - 1:
            raise ValueError("The batch is full. You need to save the current data batch before appending more data.")

        # Append the data to the current position in the batch
        for key in data_dict:
            with self.counter.get_lock():
                self.counter += 1
            self.data[key][self.counter] = data_dict[key]

    def save_as_jsonl(self, jsonl_filename=None):
        items_len = self.counter.value + 1
        # Remove the actual JSONL file from the operating system if it exists
        if os.path.exists(jsonl_filename):
            os.remove(jsonl_filename)
        
        with jsonlines.open(jsonl_filename, 'a') as f:
            for i in range(items_len):
                f.write({key: self.data[key][i] for key in self.data.keys()}, f)

        # Delete the data from the memory
        del LazyWriter.writers[self.name]
        del self

    def save(self):
        items_len = self.counter.value + 1

        for key, item in self.data.items():
            filename = f'{self.name}/{key}.npz'

            # Remove the actual file from the operating system if it exists
            if os.path.exists(filename):
                os.remove(filename)
            
            # Save data to file 
            with open(filename, 'w') as f:
                    np.savez(f, *item[:items_len])

        # Delete the data from the memory
        del LazyWriter.writers[self.name]
        del self
