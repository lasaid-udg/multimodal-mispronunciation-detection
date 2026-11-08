import numpy as np

from md_emg.multiplicity import benjamini_hochberg, parse_run_name


def test_bh_known_values():
    p = np.array([0.01, 0.04, 0.03, 0.002])
    q = benjamini_hochberg(p)
    expected = np.array([0.02, 0.04, 0.04, 0.008])
    assert np.allclose(q, expected)


def test_parse_run_names():
    g = parse_run_name('general_ch2_r10_20261019_182140_x')
    assert g['family'] == 'general' and g['channel'] == 'CH2'
    p = parse_run_name('ph_theta_ch4_r10_20261020_012259_x')
    assert p['family'] == 'phoneme_channel' and p['phoneme'] == '/θ/' and p['channel'] == 'CH4'
