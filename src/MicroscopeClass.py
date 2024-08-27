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
    psf_z: int = 130
    psf_yx: int = 130
    microscope_saving_format: str = 'pycromanager'

