from abc import abstractmethod, ABC
from typing import Optional, Any, Sequence, List
from torch.utils.data import Dataset
from utils import  ImageTxtSample

def collate_fn(batch_data: List[ImageTxtSample]):
    collate_batch_data = []
    for data in batch_data: 
        content = {
            'image_path': data["image_path"],
            'text': data["text"]
            }
        message = [
            {
                "role": "user",
                "content": content
            }
        ]

        collate_data = {
            'message': message,
        }

        collate_batch_data.append(collate_data)
    return collate_batch_data

class BaseDataset(Dataset, ABC):
    """
    Base class for datasets, __getitem__ function return Union[ImageTxtSample, TxtSample].
    """

    dataset_id: str # Identifier for the dataset
    follow_rules: bool # Whether or not follow the safety specification rules

    def __init__(self, dataset_id: str, follow_rules: bool, **kwargs) -> None:
        """
        Initializing dataset instance.
        Arguments:
            dataset_id: Identifier for the dataset
            kwargs: extra configurations
            
        """
        self.dataset_id = dataset_id
        self.follow_rules = follow_rules
        self.dataset: List[Any] = []
        
    @abstractmethod
    def __getitem__(self, index: int) -> ImageTxtSample:
        return self.dataset[index]
    
    @abstractmethod
    def __len__(self) -> int:
        return len(self.dataset)