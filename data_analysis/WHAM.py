import json
import h5py
import MDSplus as mds

from WhamData import *
from DataSpec import Spectrometer

def save_dict_to_h5(data, filename):
    with h5py.File(filename, "w") as h5file:
        def recurse(h5group, d):
            for key, value in d.items():
                if isinstance(value, dict):
                    subgroup = h5group.create_group(key)
                    recurse(subgroup, value)
                elif isinstance(value, np.ndarray):
                    h5group.create_dataset(key, data=value)
                else:
                    # Scalars or lists
                    h5group[key] = value
        recurse(h5file, data)


def load_dict_from_h5(filename):
    def recurse(h5obj):
        result = {}
        for key, item in h5obj.items():
            if isinstance(item, h5py.Dataset):
                result[key] = item[()]  # Convert to NumPy array or scalar
            elif isinstance(item, h5py.Group):
                result[key] = recurse(item)
        return result

    with h5py.File(filename, 'r') as f:
        return recurse(f)


class WHAM:
    """
    Super class that manages all WHAM diagnostics for a specific shot
    """
    def __init__(self, shot,
                       # subset for loading
                       load_list= ['bias', 'flux', 'interferometer', 
                                   'gas', 'ech', 'edge_probes', 
                                   'nbi', 'shine', 
                                   'bdot', 'bolometer', 'dalpha', 
                                   'oes'
                                  ],
                 ):
        self.shot = shot
        self.tree =  mds.Tree("wham",shot)

        # all implemented diagonstics
        diagnostic_list = {
            'bias': BiasPPS,
            'flux': FluxLoop,
            'interferometer': Interferometer,
            'axuv': AXUV,
            'gas': Gas,
            'ech': ECH,
            'edge_probes': EdgeProbes,
            'nbi': NBI,
            'shine' : ShineThrough,
            'bdot' : Bdot,
            'bolometer': Bolometer,
            'dalpha': Dalpha,
            'oes': Spectrometer,
            'endring' : EndRing
        }

        # Dictionary of diagnostic classes to instantiate
        self.diagnostic_classes = { key: diagnostic_list[key] for key in load_list }
        
        # Load each diagnostic
        for name, diag_class in self.diagnostic_classes.items():
            setattr(self, name, diag_class(shot))

    def print_diagnostic_status(self):
        """Print the status of all diagnostics"""
        print(f"Diagnostic status for shot {self.shot}:")
        for name in self.diagnostic_classes.keys():
            diag = getattr(self, name)
            status_str = "Loaded" if diag.is_loaded else f"Failed: {diag.load_status_message}"
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
        This function actually returns a big python dict

        always check status (is_loaded)
        if detail_level is sumamry, do only summary
        if detail_level is full, do summary and data
        Parameters
        ----------
        detail_level : str
            - 'status': always include is_loaded info
            - 'summary': Also include summary
            - 'full': include summary and full data
            - 'compressed': include summary and full data
    
        Returns
        -------
        dict
        """

        result = {
            "shot": self.shot,
            "is_loaded": {},
        }
    
        if detail_level in ('summary', 'full', 'compressed'):
            result["summary"] = {}
    
        if detail_level == 'full':
            result["full"] = {}

        if detail_level == 'compressed':
            result["compressed"] = {}
    
        for name in self.diagnostic_classes.keys():
            diag = getattr(self, name)
            result["is_loaded"][name] = diag.is_loaded
    
            if detail_level in ('summary', 'full', 'compressed'):
                result["summary"][name] = diag.to_dict('summary')['summary']
    
            if detail_level == 'full':
                result["full"][name] = diag.to_dict('full')['data']
    
            if detail_level == 'compressed':
                result["compressed"][name] = diag.to_dict('compressed')['compressed']

        return result

    def save_data_h5(self, filename, level='full'):
        '''
        one size fits all "level" for all diagnostics
        '''
        data = self.to_json(detail_level=level)
        save_dict_to_h5(data, filename)
        print(f"saved {filename}")

