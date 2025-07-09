import os
import pytest
from . import flowmetry

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(THIS_DIR, 'test_files')
TEST_FILES = [f for f in os.listdir(TEST_DIR) if f.endswith('.spe')]


def test_mass():
    for line in flowmetry.LINES:
        mass = flowmetry.get_mass(line)

        if line == 'OII_460':
            assert mass == flowmetry.MASS['O']
        elif line == 'OII_465':
            assert mass == flowmetry.MASS['O']
        elif line == 'CIII_465':
            assert mass == flowmetry.MASS['C']
        elif line == 'HeII_468':
            assert mass == flowmetry.MASS['He']


def load_config():
    config = flowmetry.load_config(os.path.join(THIS_DIR, 'config.ini') )
    assert 'plot_tmax' in config


@pytest.mark.parametrize("filename", TEST_FILES)
def test_run(filename):
    test_file_path = os.path.join(TEST_DIR, filename)
    flowmetry.run(test_file_path)


