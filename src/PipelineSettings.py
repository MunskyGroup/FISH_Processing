from dataclasses import dataclass, fields
import os

repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class Settings:
    return_data_to_NAS: int = 1
    NUMBER_OF_CORES: int = 4
    save_files: int = 1
    user_select_number_of_images_to_run: int = 100_000  # TODO: This is bad but I want it to select all and am too lazy
                                                        # to deal with it rn
    download_data_from_NAS: int = 0  # 0 for local, 1 for NAS
    connection_config_location: str = str(os.path.join(repo_path, 'config_nas.yml')) #r"C:\Users\Jack\Desktop\config_nas.yml" # r"/home/formanj/FISH_Processing_JF/FISH_Processing/config.yml"
    share_name: str = 'share'
    display_plots: bool = True
    load_in_mask: bool = False
    kwargs: dict = None

    def __init__(self, **kwargs):
        # Loop over all fields defined in the dataclass
        for f in fields(self):
            # Set the attribute with the value from kwargs, or the default if not provided
            setattr(self, f.name, kwargs.get(f.name, f.default))

    def __post_init__(self):
        if self.kwargs is not None:
            for key, value in self.kwargs.items():
                setattr(self, key, value)

    def pipeline_init(self):
        pass

    def to_dict(self):
        return self.__dict__
    







