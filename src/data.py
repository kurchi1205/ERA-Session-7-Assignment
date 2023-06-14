from torchvision import datasets, transforms

def get_transforms(train=False):
    if train:
        return transforms.Compose([
                            transforms.RandomRotation((-7.0, 7.0), fill=(1,)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,)) # The mean and std have to be sequences (e.g., tuples), therefore you should add a comma after the values. 
                        ])
    else:
        return transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                                ])
    
    
def get_data(train=False):
    transforms = get_transforms(train)
    if train:
        data = datasets.MNIST('./data', train=True, download=True, transform=transforms)
        return data
    else:
        data = datasets.MNIST('./data', train=False, download=True, transform=transforms)
        return data
