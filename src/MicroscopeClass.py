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
    spot_yx: int = 300
    microscope_saving_format: str = 'pycromanager'

