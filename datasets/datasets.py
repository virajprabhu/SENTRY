datasets = {}

name2benchmark = {	
                    "mnist": "MNIST", "svhn": "SVHN", \
                    "real": "DomainNet", "sketch": "DomainNet", "clipart": "DomainNet", "painting": "DomainNet", \
                    "Real_World": "OfficeHome", "Product": "OfficeHome", "Clipart": "OfficeHome", "Art": "OfficeHome", \
                    "visda_train": "VisDA2017", "visda_test": "VisDA2017"
			    }

def register_dataset(name):
    def decorator(cls):
        datasets[name] = cls
        return cls
    return decorator

def get_dataset(name, *args):
    if name not in name2benchmark: return None
    return datasets[name2benchmark[name]](*args)
