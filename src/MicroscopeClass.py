

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
                 psf_z:int = 350,
                 psf_yx:int = 130,
                 microscope_saving_format:str = 'pycromanager'
                 ) -> None:
        self.voxel_size_yx = voxel_size_yx
        self.psf_z = psf_z
        self.psf_yx = psf_yx
        self.microscope_saving_format = microscope_saving_format

    def get_voxel_size_yx(self) -> int:
        return self.voxel_size_yx
    
    def get_psf_z(self) -> int:
        return self.psf_z
    
    def get_psf_yx(self) -> int:
        return self.psf_yx
    
    def get_microscope_saving_format(self) -> str:
        return self.microscope_saving_format
    
    def get_microscope_settings(self) -> dict:
        return {'voxel_size_yx': self.voxel_size_yx,
                'psf_z': self.psf_z,
                'psf_yx': self.psf_yx,
                'microscope_saving_format': self.microscope_saving_format}
    

