from dataclasses import dataclass


@dataclass
class ScopeClass:
    """
    Class to store the parameters of the microscope.
    Attributes:
    voxel_size_yx: int
    psf_z: int
    psf_yx: int

    Default values will be for Terminator Scope
    
    """
    voxel_size_yx: int = 130
    spot_z: int = 500
    spot_yx: int = 360
    microscope_saving_format: str = 'pycromanager'
    kwargs: dict = None

    def __post_init__(self):
        if self.kwargs is not None:
            for key, value in self.kwargs.items():
                setattr(self, key, value)

    def pipeline_init(self):
        pass

    def to_dict(self):
        return self.__dict__
    

