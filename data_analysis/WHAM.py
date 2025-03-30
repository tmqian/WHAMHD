import json
from WhamData import *
from DataSpec import Spectrometer

class WHAM:
    """
    Super class that manages all WHAM diagnostics for a specific shot
    """
    def __init__(self, shot):
        self.shot = shot

        #self.radiation = Radiation(shot)
        #self.gas = Gas(shot)
        #self.ion_probe = IonProbe(shot)
        #self.end_ring = EndRing(shot)

        # Dictionary of diagnostic classes to instantiate
        self.diagnostic_classes = {
            'bias': BiasPPS,
            'flux': FluxLoop,
            'interferometer': Interferometer,
            #'axuv': AXUV,
            'ech': ECH,
            #'edge_probes': EdgeProbes,
            'nbi': NBI,
            'shine' : ShineThrough,
            'bdot' : Bdot,
            #'bolometer': Bolometer,
            #'dalpha': Dalpha,
            #'oes': Spectrometer
        }
        
        # Load each diagnostic
        for name, diag_class in self.diagnostic_classes.items():
            setattr(self, name, diag_class(shot))

    def print_diagnostic_status(self):
        """Print the status of all diagnostics"""
        print(f"Diagnostic status for shot {self.shot}:")
        for name in self.diagnostic_classes.keys():
            diag = getattr(self, name)
            status_str = "✓ Loaded" if diag.is_loaded else f"✗ Failed: {diag.load_status_message}"
            print(f"  {name}: {status_str}")

    def to_dict(self, detail_level='summary'):
        """
        Create a dictionary summary of the shot data
        
        Parameters:
        -----------
        detail_level : str
            Level of detail to include:
            - 'status': Only loading status
            - 'summary': Key metrics (default)
            - 'full': Complete dataset
        """
        summary = {
            "shot": self.shot,
            "diagnostics": {}
        }
        
        # Add data from all diagnostics with the specified detail level
        for name in self.diagnostic_classes.keys():
            diag = getattr(self, name)
            summary["diagnostics"][name] = diag.to_dict(detail_level)
        
        return summary

    def to_json(self, detail_level='status', indent=None):
        """
        Convert the summary dictionary to a JSON string
        """
        
        if detail_level == 'status':
            return {
                "shot": self.shot,
                "is_loaded": {
                    name: getattr(self, name).is_loaded
                    for name in self.diagnostic_classes.keys()
                }
            }

        # Regular format for other detail levels
        summary = {
            "shot": self.shot,
            "diagnostics": {}
        }
    
        # Add data from all diagnostics with the specified detail level
        for name in self.diagnostic_classes.keys():
            diag = getattr(self, name)
            summary["diagnostics"][name] = diag.to_dict(detail_level)
    
        return summary

