class ScopeClass():
    """
    Class to store the parameters of the microscope.
    Attributes:
    voxel_size_yx: int
    psf_z: int
    psf_yx: int

    Default values will be for Terminator Scope
    
    """

    def __init__(self,
                 voxel_size_yx:int = 130,
                 psf_z:int = 500,
                 psf_yx:int = 130*3,
                 microscope_saving_format:str = 'pycromanager'
                 ) -> None:
        self.voxel_size_yx = voxel_size_yx
        self.psf_z = psf_z
        self.psf_yx = psf_yx
        self.microscope_saving_format = microscope_saving_format
