import dataclasses
from loader import DatasetLoader


@dataclasses.dataclass
class Datasetlabeler:
    dataset_loader: DatasetLoader
    overwrite_existing: bool = False

    def label(self, dataset):
        for item in dataset:
            item['label'] = self.dataset_loader.get_label(item['id'])
        return dataset
