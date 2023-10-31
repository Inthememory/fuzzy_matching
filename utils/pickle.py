import pickle


def pickle_save(obj, path_output: str):
    pickle.dump(obj, open(f"{path_output}.pickle", "wb"))


def pickle_get(path_output: str):
    return pickle.load(open(f"{path_output}.pickle", "rb"))
