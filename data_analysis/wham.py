import json
from WhamData import *
from DataSpec import Spectrometer

class WHAM:
    """
    Super class that manages all WHAM diagnostics for a specific shot
    """
    def __init__(self, shot):
        self.shot = shot

        #self.gas = Gas(shot)
        #self.radiation = Radiation(shot)
        #self.gas = Gas(shot)
        #self.ion_probe = IonProbe(shot)
        #self.end_ring = EndRing(shot)

        # Dictionary of diagnostic classes to instantiate
        self.diagnostic_classes = {
            'bias': BiasPPS,
            'interferometer': Interferometer,
            'flux': FluxLoop,
            'axuv': AXUV,
            'ech': ECH,
            'edge_probes': EdgeProbes,
            'nbi': NBI,
            'shine' : ShineThrough,
            'bolometer': Bolometer,
            'dalpha': Dalpha,
            'oes': Spectrometer
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

shot = 250326066
wham = WHAM(shot)
import pdb
pdb.set_trace()
