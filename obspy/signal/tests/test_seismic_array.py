#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The SeismicArray test suite.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import io
import os
import unittest

import numpy as np
import matplotlib.pyplot as plt

from obspy import Stream, Trace, UTCDateTime
from obspy.core.util.testing import ImageComparison
from obspy.signal.array_analysis import SeismicArray
from obspy.signal.array_analysis.seismic_array import _get_stream_offsets
from obspy.signal.util import util_lon_lat
from obspy import read, read_events
from obspy.core.inventory import read_inventory
from obspy.core.inventory.channel import Channel
from obspy.core.inventory.station import Station
from obspy.core.inventory.network import Network
from obspy.core.inventory.inventory import Inventory

KM_PER_DEG = 111.1949

class SeismicArrayTestCase(unittest.TestCase):
    """
    Test cases for array and array analysis functions.
    """

    def setUp(self):
        self.path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                 'data'))
        self.path_images = os.path.join(os.path.dirname(__file__), 'images')

        def create_simple_array(coords, sys='xy', include_cha=False):
            """
            Set up a legal array more easily from x-y or long-lat coordinates.
            Note it's usually lat-lon in other applications.
            """
            if sys == 'xy':
                coords_lonlat = [list(util_lon_lat(0, 0, stn[0], stn[1]))
                                 for stn in coords]
                for i, cor in enumerate(coords_lonlat):
                    cor.append(coords[i][2])
            else:
                coords_lonlat = coords

            if include_cha:
                chas = [Channel(str(_i), str(_i), coords_lonlat[_i][1],
                                coords_lonlat[_i][0],
                                coords_lonlat[_i][2] * 1000, 0)
                        for _i in range(len(coords_lonlat))]
                stns = [Station(str(_i), coords_lonlat[_i][1],
                                coords_lonlat[_i][0],
                                coords_lonlat[_i][2] * 1000,
                                channels=[chas[_i]])
                        for _i in range(len(coords_lonlat))]
            else:
                stns = [Station(str(_i), coords_lonlat[_i][1],
                                coords_lonlat[_i][0],
                                coords_lonlat[_i][2] * 1000)
                        for _i in range(len(coords_lonlat))]
            testinv = Inventory([Network("testnetwork", stations=stns)],
                                'testsender')
            return SeismicArray('testarray', inventory=testinv)

        # Set up an array for geometry tests.
        geometry_coords = [[0, 0, 0], [2, 0, 0], [1, 1, 0],
                           [0, 2, 0], [2, 2, 0]]
        self.geometry_array = create_simple_array(geometry_coords, 'longlat')
        self.geometry_array_w_chas = create_simple_array(geometry_coords,
                                                         'longlat',
                                                         include_cha=True)
        # Set up simple array with heights to test 3D correction
        geometry_coords_w_h = [[0, 0, 0], [2, 0, 1], [1, 1, 0.5],
                               [0, 2, 0], [2, 2, 1]]
        self.array_3d = create_simple_array(geometry_coords_w_h,
                                            'longlat')

        # Set up the test array for the _covariance_array_processing,
        # stream_offset and array_rotation_strain tests.
        self.fk_testcoords = np.array([[0.0, 0.0, 0.0],
                                       [-5.0, 7.0, 0.0],
                                       [5.0, 7.0, 0.0],
                                       [10.0, 0.0, 0.0],
                                       [5.0, -7.0, 0.0],
                                       [-5.0, -7.0, 0.0],
                                       [-10.0, 0.0, 0.0]])
        self.fk_array = create_simple_array(self.fk_testcoords)

        # Set up test array for transff tests.
        transff_testcoords = np.array([[10., 60., 0.],
                                       [200., 50., 0.],
                                       [-120., 170., 0.],
                                       [-100., -150., 0.],
                                       [30., -220., 0.]])
        transff_testcoords /= 1000.
        self.transff_array = create_simple_array(transff_testcoords, 'longlat')

        # setup for rotation testing
        array_coords = np.array([[0.0, 0.0, 0.0],
                                 [-5.0, 7.0, 0.0],
                                 [5.0, 7.0, 0.0],
                                 [10.0, 0.0, 0.0],
                                 [5.0, -7.0, 0.0],
                                 [-5.0, -7.0, 0.0],
                                 [-10.0, 0.0, 0.0]])

        stations = []
        for i, cor in enumerate(array_coords):
            sta = Station('S%d' % i, cor[0], cor[1], cor[2],
                          channels=[Channel('C%dN' % i, '', cor[0], cor[1],
                                            cor[2], 0),
                                    Channel('C%dE' % i, '', cor[0], cor[1],
                                            cor[2], 0),
                                    Channel('C%dZ' % i, '', cor[0], cor[1],
                                            cor[2], 0)])
            stations.append(sta)

        rotnet = Network('N', stations=stations)
        rotinv = Inventory(networks=[rotnet])
        self.rotatation_array = SeismicArray(name='Rot_Array', inventory=rotinv)

        # array_coords_km is needed to compute the test results, array_coords
        # is used as input for the derive_rotation_from_array method
        self.rotatation_array_coords_km = \
            self.rotatation_array._geometry_dict_to_array(
            self.rotatation_array._get_geometry_xyz(0, 0, 0))[::3]
        self.rotatation_array_coords = \
            self.rotatation_array._geometry_dict_to_array(
            self.rotatation_array.geometry)[::3]

        ts1 = np.empty((1000, 7))
        ts2 = np.empty((1000, 7))
        ts3 = np.empty((1000, 7))
        ts1.fill(np.NaN)
        ts2.fill(np.NaN)
        ts3.fill(np.NaN)
        self.rotation_ts = (ts1, ts2, ts3)
        # parameters: (sigmau, vp, vs)
        self.rotatation_parameters = (0.0001, 1.93, 0.326)

    def test__get_geometry(self):
        geo = self.geometry_array.geometry
        geo_w_chas = self.geometry_array_w_chas.geometry
        geo_exp = {'testnetwork.0..': {'absolute_height_in_km': 0.0,
                                       'latitude': 0.0, 'longitude': 0.0},
                   'testnetwork.1..': {'absolute_height_in_km': 0.0,
                                       'latitude': 0.0, 'longitude': 2.0},
                   'testnetwork.2..': {'absolute_height_in_km': 0.0,
                                       'latitude': 1.0, 'longitude': 1.0},
                   'testnetwork.3..': {'absolute_height_in_km': 0.0,
                                       'latitude': 2.0, 'longitude': 0.0},
                   'testnetwork.4..': {'absolute_height_in_km': 0.0,
                                       'latitude': 2.0, 'longitude': 2.0}}
        geo_exp_ch = {'testnetwork.0.0.0': {'absolute_height_in_km': 0.0,
                                            'latitude': 0.0, 'longitude': 0.0},
                      'testnetwork.1.1.1': {'absolute_height_in_km': 0.0,
                                            'latitude': 0.0, 'longitude': 2.0},
                      'testnetwork.2.2.2': {'absolute_height_in_km': 0.0,
                                            'latitude': 1.0, 'longitude': 1.0},
                      'testnetwork.3.3.3': {'absolute_height_in_km': 0.0,
                                            'latitude': 2.0, 'longitude': 0.0},
                      'testnetwork.4.4.4': {'absolute_height_in_km': 0.0,
                                            'latitude': 2.0, 'longitude': 2.0}}

        self.assertEqual(geo, geo_exp)
        self.assertEqual(geo_w_chas, geo_exp_ch)

    def test_get_geometry_xyz(self):
        """
        Test _get_geometry_xyz and, implicitly, _get_geometry (necessary because
        self.geometry is a property and can't be set).
        """
        geox_exp = {'testnetwork.0..': {'x': -111.31564682647114,
                                        'y': -110.5751633754653, 'z': 0.0},
                    'testnetwork.1..': {'x': 111.31564682647114,
                                        'y': -110.5751633754653, 'z': 0.0},
                    'testnetwork.2..': {'x': 0.0, 'y': 0.0, 'z': 0.0},
                    'testnetwork.3..': {'x': -111.28219117308639, 'y':
                                        110.5751633754653, 'z': 0.0},
                    'testnetwork.4..': {'x': 111.28219117308639, 'y':
                                        110.5751633754653, 'z': 0.0}}
        geoxno3d = self.geometry_array._get_geometry_xyz(1, 1, 0,
                                                         correct_3dplane=False)
        geox = self.geometry_array._get_geometry_xyz(1, 1, 0,
                                                     correct_3dplane=True)
        # For flat array:
        self.assertEqual(geoxno3d, geox)
        # Use almost equal as calculations appear to be imprecise on OS X.
        for key_outer in geox_exp:
            for key_inner in geox_exp[key_outer]:
                self.assertAlmostEqual(geox[key_outer][key_inner],
                                       geox_exp[key_outer][key_inner])

    def test_center_of_gravity(self):
        self.assertEqual(self.geometry_array.center_of_gravity,
                         {'absolute_height_in_km': 0.0,
                          'latitude': 1.0, 'longitude': 1.0})

    def test_geometrical_center(self):
        self.assertEqual(self.geometry_array.geometrical_center,
                         {'absolute_height_in_km': 0.0,
                          'latitude': 1.0, 'longitude': 1.0})

    def test_apertur(self):
        self.assertAlmostEqual(self.geometry_array.aperture, 313.7, 0)

    def test_extend(self):
        extend = {'min_latitude': 0.0, 'max_latitude': 2.0,
                  'min_longitude': 0.0, 'max_longitude': 2.0,
                  'min_absolute_height_in_km': 0.0,
                  'max_absolute_height_in_km': 0.0}
        self.assertTrue(self.geometry_array.extent, extend)

    def test__coordinate_values(self):
        coordinate_lists = ([0.0, 0.0, 1.0, 2.0, 2.0],
                            [0.0, 2.0, 1.0, 0.0, 2.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0])
        self.assertTrue(self.geometry_array._coordinate_values(),
                        coordinate_lists)

    def test__get_timeshift_baz(self):
        # test 2D calculation
        ref_table_2d = {None: np.array([-2, -1,  0,  1,  2]),
                        'testnetwork.0..':
                            np.array([-2.82208081, -1.4110404,
                                      -0., 1.4110404, 2.82208081]),
                        'testnetwork.1..':
                            np.array([0.00941771, 0.00470886,
                                      0., -0.00470886, -0.00941771]),
                        'testnetwork.2..': np.array([0.,  0.,  0., -0., -0.]),
                        'testnetwork.3..':
                            np.array([-0.00899221, -0.00449611,
                                      -0., 0.00449611, 0.00899221]),
                        'testnetwork.4..':
                            np.array([2.82165531, 1.41082765,
                                      0., -1.41082765, -2.82165531])}

        tshft_table_2d = self.geometry_array._get_timeshift_baz(-2, 2, 1, 45,
                                                     latitude=1.0,
                                                     longitude=1.0,
                                                     absolute_height_in_km=0.0,
                                                     static3d=False,
                                                     vel_cor=4.0)

        for key, value in list(tshft_table_2d.items()):
            self.assertTrue(np.allclose(value, ref_table_2d[key]))

        # test 2D calculation
        ref_table_3d = {None: np.array([-2, -1, 0, 1, 2]),
                        'testnetwork.0..':
                            np.array([-2.69740474, -1.28612131, 0.125,
                                      1.5359595, 2.94675688]),
                        'testnetwork.1..':
                            np.array([-0.11525835, -0.12021024, -0.125,
                                      -0.12962795, -0.13409378]),
                        'testnetwork.2..':
                            np.array([0.,  0.,  0.,  0.,  0.]),
                        'testnetwork.3..':
                            np.array([0.11568385, 0.12042299, 0.125,
                                      0.1294152, 0.13366828]),
                        'testnetwork.4..':
                            np.array([2.69697924, 1.28590856, -0.125,
                                      -1.53574675, -2.94633138])}
        tshft_table_3d = self.array_3d._get_timeshift_baz(-2, 2, 1, 45,
                                                          latitude=1.0,
                                                          longitude=1.0,
                                                          absolute_height_in_km=0.5,
                                                          static3d=True,
                                                          vel_cor=4.0)
        for key, value in list(tshft_table_3d.items()):
            self.assertTrue(np.allclose(value, ref_table_3d[key]))

    def test__get_timeshift(self):
        ref_table_2d = [[[-1.9955125, -2.989939, -3.9843657],
                         [-2.9965985, -3.991025, -4.9854517],
                         [-3.9976842, -4.9921107, -5.9865375]],
                        [[0.00665933, -0.9877672, -1.9821938],
                         [1.0077453, 0.01331866, -0.9811079],
                         [2.008831, 1.0144045, 0.01997799]],
                        [[0., 0., 0.],
                         [0., 0., 0.],
                         [0., 0., 0.]],
                        [[-0.00635846, 0.9880681, 1.9824947],
                         [-1.0071435, -0.01271691, 0.98170966],
                         [-2.0079286, -1.0135019, -0.01907537]],
                        [[1.9952116, 2.989638, 3.9840648],
                         [2.9959967, 3.9904232, 4.98485],
                         [3.9967816, 4.991208, 5.985635]]]
        tshft_table_2d = self.geometry_array._get_timeshift(1, 1, 1, 3, 3,
                                                            latitude=None,
                                                            longitude=None,
                                                            absolute_height=None,
                                                            vel_cor=4.,
                                                            static3d=False)
        self.assertTrue(np.allclose(ref_table_2d, tshft_table_2d))

        ref_table_3d = [[[-1.8706744, -2.865344, -3.860177],
                         [-2.8720033, -3.8666737, -4.8615074],
                         [-3.8734956, -4.868167, -5.863002]],
                        [[-0.11817881, -1.1123621, -2.1063824],
                         [0.8831503, -0.11103263, -1.105052],
                         [1.8846426,  0.89046043, -0.10355763]],
                        [[0.,  0.,  0.],
                         [0.,  0.,  0.],
                         [0.,  0.,  0.]],
                        [[0.11847968,  1.112663,  2.1066833],
                         [-0.8825485,  0.11163438,  1.1056538],
                         [-1.88374, -0.88955784,  0.10446025]],
                        [[1.8703735,  2.8650432,  3.8598762],
                         [2.8714018,  3.866072,  4.8609056],
                         [3.8725932,  4.8672643,  5.862099 ]]]
        tshft_table_3d = self.array_3d._get_timeshift(1, 1, 1, 3, 3,
                                                      latitude=None,
                                                      longitude=None,
                                                      absolute_height=None,
                                                      vel_cor=4.,
                                                      static3d=True)
        self.assertTrue(np.allclose(ref_table_3d, tshft_table_3d))

    def test_inventory_cull(self):
        time = UTCDateTime('2016-04-05T06:44:0.0Z')
        # Method should work even when traces do not cover same times.
        st = Stream([
            Trace(data=np.empty(20), header={'network': 'BP', 'station':
                  'CCRB', 'location': '1', 'channel': 'BP1',
                    'starttime': time}),
            Trace(data=np.empty(20), header={'network': 'BP', 'station':
                  'EADB', 'channel': 'BPE', 'starttime': time-60})])
        # Set up channels, correct ones first. The eadb channel should also be
        # selected despite no given time.
        kwargs = {'latitude': 0, 'longitude': 0, 'elevation': 0, 'depth': 0}
        ch_ccrb = Channel(code='BP1', start_date=time-10, end_date=time+60,
                          location_code='1', **kwargs)
        wrong = [Channel(code='BP1', start_date=time-60, end_date=time-10,
                         location_code='1', **kwargs),
                 Channel(code='BP2', location_code='1', **kwargs)]
        ccrb = Station('CCRB', 0, 0, 0, channels=[ch_ccrb] + wrong)

        ch_eadb = Channel(code='BPE', location_code='', **kwargs)
        wrong = Channel(code='BPE', location_code='2', **kwargs)
        eadb = Station('EADB', 0, 0, 0, channels=[ch_eadb, wrong])
        wrong_stn = Station('VCAB', 0, 0, 0, channels=[ch_eadb, wrong])

        array = SeismicArray('testarray', Inventory([Network('BP',
                             stations=[ccrb, eadb, wrong_stn])], 'testinv'))

        array.inventory_cull(st)
        self.assertEqual(array.inventory[0][0][0], ch_ccrb)
        self.assertEqual(array.inventory[0][1][0], ch_eadb)
        tbc = [array.inventory.networks, array.inventory[0].stations,
               array.inventory[0][0].channels, array.inventory[0][1].channels]
        self.assertEqual([len(item) for item in tbc], [1, 2, 1, 1])

    def test_covariance_array_processing(self):
        # Generate some synthetic data for the FK/Capon tests
        np.random.seed(2348)
        slowness = 1.3       # in s/km
        baz_degree = 20.0    # 0.0 > source in x direction
        baz = baz_degree * np.pi / 180.
        df = 100             # samplerate
        # SNR = 100.         # signal to noise ratio
        amp = .00001         # amplitude of coherent wave
        length = 500         # signal length in samples
        coherent_wave = amp * np.random.randn(length)
        # time offsets in samples

        # here is something wrong with the units:
        # with /KM_PER_DEG it is at least close
        self.fk_testcoords /= KM_PER_DEG

        dt = np.round(df * slowness * (np.cos(baz) * self.fk_testcoords[:, 1] +
                                       np.sin(baz) * self.fk_testcoords[:, 0]))
        trl = []
        for i in range(len(self.fk_testcoords)):
            tr = Trace(coherent_wave[int(-(np.min(dt) - 1) + dt[i]):
                                     int(-(np.max(dt) + 1) + dt[i])].copy())
            tr.stats.sampling_rate = df
            # lowpass random signal to f_nyquist / 2
            tr.filter("lowpass", freq=df / 4.)
            trl.append(tr)
        st = Stream(trl)
        stime = UTCDateTime(1970, 1, 1, 0, 0)
        fk_args = (st, 2, 0.2, -3, 3, -3, 3, 0.1,
                   -1e99, -1e99, 1, 8, stime, stime + 4)
        # Tests for FK analysis:
        out = self.fk_array._covariance_array_processing(*fk_args,
                                                         prewhiten=0, method=0)
        raw = """
        9.68742255e-01 1.95739086e-05 1.84349488e+01 1.26491106e+00
        9.60822403e-01 1.70468277e-05 1.84349488e+01 1.26491106e+00
        9.61689241e-01 1.35971034e-05 1.84349488e+01 1.26491106e+00
        9.64670470e-01 1.35565806e-05 1.84349488e+01 1.26491106e+00
        9.56880885e-01 1.16028992e-05 1.84349488e+01 1.26491106e+00
        9.49584782e-01 9.67131311e-06 1.84349488e+01 1.26491106e+00
        """
        ref = np.loadtxt(io.StringIO(raw), dtype=np.float32)
        self.assertTrue(np.allclose(ref, np.array(out[:, 1:], dtype=float),
                                    rtol=1e-6))

        out = self.fk_array._covariance_array_processing(*fk_args,
                                                         prewhiten=1, method=0)
        raw = """
        1.40997967e-01 1.95739086e-05 1.84349488e+01 1.26491106e+00
        1.28566503e-01 1.70468277e-05 1.84349488e+01 1.26491106e+00
        1.30517975e-01 1.35971034e-05 1.84349488e+01 1.26491106e+00
        1.34614854e-01 1.35565806e-05 1.84349488e+01 1.26491106e+00
        1.33609938e-01 1.16028992e-05 1.84349488e+01 1.26491106e+00
        1.32638966e-01 9.67131311e-06 1.84349488e+01 1.26491106e+00
        """
        ref = np.loadtxt(io.StringIO(raw), dtype=np.float32)
        self.assertTrue(np.allclose(ref, np.array(out[:, 1:], dtype=float)))

        # Tests for Capon
        out = self.fk_array._covariance_array_processing(*fk_args,
                                                         prewhiten=0, method=1)
        raw = """
        9.06938200e-01 9.06938200e-01  1.49314172e+01  1.55241747e+00
        8.90494375e+02 8.90494375e+02 -9.46232221e+00  1.21655251e+00
        3.07129784e+03 3.07129784e+03 -4.95739213e+01  3.54682957e+00
        5.00019137e+03 5.00019137e+03 -1.35000000e+02  1.41421356e-01
        7.94530414e+02 7.94530414e+02 -1.65963757e+02  2.06155281e+00
        6.08349575e+03 6.08349575e+03  1.77709390e+02  2.50199920e+00
        """
        ref = np.loadtxt(io.StringIO(raw), dtype=np.float32)
        self.assertTrue(np.allclose(ref, np.array(out[:, 1:], dtype=float),
                                    rtol=1e-6))

        out = self.fk_array._covariance_array_processing(*fk_args,
                                                         prewhiten=1, method=1)
        raw = """
        1.30482688e-01 9.06938200e-01  1.49314172e+01  1.55241747e+00
        8.93029978e-03 8.90494375e+02 -9.46232221e+00  1.21655251e+00
        9.55393634e-03 1.50655072e+01  1.42594643e+02  2.14009346e+00
        8.85762420e-03 7.27883670e+01  1.84349488e+01  1.26491106e+00
        1.51510617e-02 6.54541771e-01  6.81985905e+01  2.15406592e+00
        3.10761699e-02 7.38667657e+00  1.13099325e+01  1.52970585e+00
        """
        ref = np.loadtxt(io.StringIO(raw), dtype=np.float32)
        self.assertTrue(np.allclose(ref, np.array(out[:, 1:], dtype=float),
                                    rtol=1e-6))

    def test_get_stream_offset(self):
        """
        Test case for #682
        """
        stime = UTCDateTime(1970, 1, 1, 0, 0)
        etime = UTCDateTime(1970, 1, 1, 0, 0) + 10
        data = np.empty(20)
        # sampling rate defaults to 1 Hz
        st = Stream([
            Trace(data, {'starttime': stime - 1}),
            Trace(data, {'starttime': stime - 4}),
            Trace(data, {'starttime': stime - 2}),
        ])
        spoint, epoint = _get_stream_offsets(st, stime, etime)
        self.assertTrue(np.allclose([1, 4, 2], spoint))
        self.assertTrue(np.allclose([8, 5, 7], epoint))

    def test_fk_array_transff_freqslowness(self):
        transff = self.transff_array.array_transfer_function_freqslowness(40, 20, 1,
                                                                          10, 1)
        # had to be changed because the x-y to lat lon conversion is different
        # in the older test_sonic file which includes the test for the
        # old array methods
        transffth = np.array(
            [[0.41915119, 0.25248452, 0.24751548, 0.33333333, 0.67660475],
             [0.33333333, 0.41418215, 0.25248452, 0.65672859, 0.24751548],
             [0.32339525, 0.34327141, 1.00000000, 0.34327141, 0.32339525],
             [0.24751548, 0.65672859, 0.25248452, 0.41418215, 0.33333333],
             [0.67660475, 0.33333333, 0.24751548, 0.25248452, 0.41915119]])
        # transff is normalized, normalize transffth as well
        transffth /= np.max(transffth)
        np.testing.assert_array_almost_equal(transff, transffth, decimal=6)

    def test_fk_array_transff_wavenumber(self):
        transff = self.transff_array.array_transfer_function_wavenumber(40, 20)
        # had to be changed because the x-y to lat lon conversion is different
        # in the older test_sonic file which includes the test for the
        # old array methods
        transffth = np.array(
            [[3.13360360e-01, 2.98941684e-01, 1.26523678e-01,
              5.57078203e-01, 8.16891615e-04],
             [4.23775796e-02, 2.47377842e-01, 2.91010683e-01,
              6.84732871e-02, 4.80470652e-01],
             [6.73650243e-01, 9.96352135e-02, 1.00000000e+00,
              9.96352135e-02, 6.73650243e-01],
             [4.80470652e-01, 6.84732871e-02, 2.91010683e-01,
              2.47377842e-01, 4.23775796e-02],
             [8.16891615e-04, 5.57078203e-01, 1.26523678e-01,
              2.98941684e-01, 3.13360360e-01]])
        # transff is normalized, normalize transffth as well
        transffth /= np.max(transffth)
        np.testing.assert_array_almost_equal(transff, transffth, decimal=6)

    def test_three_component_beamforming(self):
        """
        Integration test for three-component beamforming with instaseis data
        and the real Parkfield array. Parameter values are fairly arbitrary.
        """
        pfield = SeismicArray('pfield', inventory=read_inventory(
                os.path.join(self.path, 'pfield_inv_for_instaseis.xml'),
                format='stationxml'))
        vel = read(os.path.join(self.path, 'pfield_instaseis.mseed'))
        out = pfield.three_component_beamforming(
            vel.select(channel='BXN'), vel.select(channel='BXE'),
            vel.select(channel='BXZ'), 64, 0, 0.6, 0.03, wavetype='P',
            freq_range=[0.1, .3], whiten=True, coherency=False)
        self.assertEqual(out.max_pow_baz, 246)
        self.assertEqual(out.max_pow_slow, 0.3)
        np.testing.assert_array_almost_equal(out.max_rel_power, 1.22923997,
                                             decimal=8)

    def test_derive_rotation_from_array(self):
        # Setup
        array = self.rotatation_array

        # array_coords_km is needed to compute the test results, array_coords
        # is used as input for the derive_rotation_from_array method
        array_coords_km = self.rotatation_array_coords_km
        array_coords = self.rotatation_array_coords

        ts1, ts2, ts3 = self.rotation_ts
        sigmau, vp, vs = self.rotatation_parameters

        # Tests function array_rotation_strain with synthetic data with pure
        # rotation and no strain

        rotx = 0.00001 * np.exp(-1 * np.square(np.linspace(-2, 2, 1000))) * \
               np.sin(np.linspace(-30 * np.pi, 30 * np.pi, 1000))
        roty = 0.00001 * np.exp(-1 * np.square(np.linspace(-2, 2, 1000))) * \
               np.sin(np.linspace(-20 * np.pi, 20 * np.pi, 1000))
        rotz = 0.00001 * np.exp(-1 * np.square(np.linspace(-2, 2, 1000))) * \
               np.sin(np.linspace(-10 * np.pi, 10 * np.pi, 1000))

        for stat in range(7):
            for t in range(1000):
                ts1[t, stat] = -1. * array_coords_km[stat, 1] * rotz[t]
                ts2[t, stat] = array_coords_km[stat, 0] * rotz[t]
                ts3[t, stat] = array_coords_km[stat, 1] * rotx[t] - \
                               array_coords_km[stat, 0] * roty[t]

        traces = []
        for i, cor in enumerate(array_coords):
            traces.append(Trace(ts2[:, i], header={'network': 'N',
                                                   'location': '',
                                                   'station': 'S%d' % i,
                                                   'channel': 'C%dN' % i}))
            traces.append(Trace(ts3[:, i], header={'network': 'N',
                                                   'location': '',
                                                   'station': 'S%d' % i,
                                                   'channel': 'C%dE' % i}))
            traces.append(Trace(ts1[:, i], header={'network': 'N',
                                                   'location': '',
                                                   'station': 'S%d' % i,
                                                   'channel': 'C%dZ' % i}))
        st = Stream(traces)

        st_out, out = array.derive_rotation_from_array(st, vp, vs, sigmau,
                                                       0, 0, 0)

        # test for equality
        np.testing.assert_array_almost_equal(rotx, st_out[0].data, decimal=12)
        np.testing.assert_array_almost_equal(roty, st_out[1].data, decimal=12)
        np.testing.assert_array_almost_equal(rotz, st_out[2].data, decimal=12)
        np.testing.assert_array_almost_equal(np.zeros(1000), out['ts_s'],
                                             decimal=15)
        np.testing.assert_array_almost_equal(np.zeros(1000), out['ts_d'],
                                             decimal=15)
        np.testing.assert_array_almost_equal(np.zeros(1000), out['ts_m'],
                                             decimal=12)

        # Tests function array_rotation_strain with synthetic data with pure
        # dilation and no rotation or shear strain

        eta = 1 - 2 * vs ** 2 / vp ** 2

        dilation = .00001 * np.exp(
            -1 * np.square(np.linspace(-2, 2, 1000))) * \
                   np.sin(np.linspace(-40 * np.pi, 40 * np.pi, 1000))

        for stat in range(7):
            for t in range(1000):
                ts1[t, stat] = array_coords_km[stat, 0] * dilation[t]
                ts2[t, stat] = array_coords_km[stat, 1] * dilation[t]
                ts3[t, stat] = array_coords_km[stat, 2] * dilation[t]

        st_out, out = array.derive_rotation_from_array(st, vp, vs, sigmau,
                                                    0, 0, 0)

        # remember free surface boundary conditions!
        #         # see Spudich et al, 1995, (A2)
        np.testing.assert_array_almost_equal(dilation * (2 - 2 * eta),
                                             out['ts_d'], decimal=12)
        np.testing.assert_array_almost_equal(dilation * 2, out['ts_dh'],
                                             decimal=12)
        np.testing.assert_array_almost_equal(
            abs(dilation * .5 * (1 + 2 * eta)), out['ts_s'], decimal=12)
        np.testing.assert_array_almost_equal(np.zeros(1000), out['ts_sh'],
                                             decimal=12)
        np.testing.assert_array_almost_equal(np.zeros(1000), st_out[0],
                                             decimal=15)
        np.testing.assert_array_almost_equal(np.zeros(1000), st_out[1],
                                             decimal=15)
        np.testing.assert_array_almost_equal(np.zeros(1000), st_out[2],
                                             decimal=15)
        np.testing.assert_array_almost_equal(np.zeros(1000), out['ts_m'],
                                             decimal=12)

        # Tests function array_rotation_strain with synthetic data with pure
        # horizontal shear strain, no rotation or dilation.

        shear_strainh = .00001 * np.exp(
            -1 * np.square(np.linspace(-2, 2, 1000))) * \
                        np.sin(np.linspace(-10 * np.pi, 10 * np.pi, 1000))

        ts3 = np.zeros(1000)
        for tr_z in st[1::3]:
            tr_z.data = np.zeros(1000)

        for stat in range(7):
            for t in range(1000):
                ts1[t, stat] = array_coords_km[stat, 1] * shear_strainh[t]
                ts2[t, stat] = array_coords_km[stat, 0] * shear_strainh[t]

        st_out, out = array.derive_rotation_from_array(st, vp, vs, sigmau,
                                                       0, 0, 0)

        np.testing.assert_array_almost_equal(np.zeros(1000), out['ts_d'],
                                             decimal=12)
        np.testing.assert_array_almost_equal(np.zeros(1000), out['ts_dh'],
                                             decimal=12)
        np.testing.assert_array_almost_equal(abs(shear_strainh), out['ts_s'],
                                             decimal=12)
        np.testing.assert_array_almost_equal(abs(shear_strainh), out['ts_sh'],
                                             decimal=12)
        np.testing.assert_array_almost_equal(np.zeros(1000), st_out[0],
                                             decimal=12)
        np.testing.assert_array_almost_equal(np.zeros(1000), st_out[1],
                                             decimal=12)
        np.testing.assert_array_almost_equal(np.zeros(1000), st_out[2],
                                             decimal=12)
        np.testing.assert_array_almost_equal(np.zeros(1000), out['ts_m'],
                                             decimal=12)

    def test__geometry_dict_to_array(self):
        geo_dict_latlon = {'1': {'latitude': 5, 'longitude': 10,
                           'absolute_height_in_km': 1}}
        geo_dict_xyz = {'1': {'x': 5, 'y': 10, 'z': 1}}

        array_latlon = self.geometry_array._geometry_dict_to_array(
            geo_dict_latlon)
        array_xyz = self.geometry_array._geometry_dict_to_array(
            geo_dict_xyz)

        ref_array = [[5., 10., 1.]]

        np.testing.assert_array_equal(array_latlon, ref_array)
        np.testing.assert_array_equal(array_xyz, ref_array)

    def test_array_plotting(self):
        arr = SeismicArray('pfield', inventory=read_inventory(
            os.path.join(self.path, 'pfield_inv_for_instaseis.xml')))
        with ImageComparison(self.path_images, "seismic_array_map.png") as ic:
            arr.plot()
            plt.savefig(ic.name)

    def test_plot_radial_transfer_function(self):
        arr = SeismicArray('pfield', inventory=read_inventory(
            os.path.join(self.path, 'pfield_inv_for_instaseis.xml')))
        with ImageComparison(self.path_images, "radialtransferfunc.png") as ic:
            arr.plot_radial_transfer_function(0, 0.6, 0.05, [0.2])
            plt.savefig(ic.name)

    def test_plot_transfer_function_wavenumber(self):
        arr = SeismicArray('pfield', inventory=read_inventory(
            os.path.join(self.path, 'pfield_inv_for_instaseis.xml')))
        with ImageComparison(self.path_images, "transfer_func_k.png") as ic:
            arr.plot_transfer_function_wavenumber(10, 0.1)
            plt.savefig(ic.name)

    def test_plot_transfer_function_freqslowness(self):
        arr = SeismicArray('pfield', inventory=read_inventory(
            os.path.join(self.path, 'pfield_inv_for_instaseis.xml')))
        with ImageComparison(self.path_images, "transfer_func_fs.png") as ic:
            arr.plot_transfer_function_freqslowness(slim=10, sstep=0.5,
                                                    freq_min=0.1, freq_max=4,
                                                    freq_step=0.2)
            plt.savefig(ic.name)
    #     """
    #     Tests the plotting of radial array transfer functions.
    #     """
    #     arr = SeismicArray('pfield', inventory=read_inventory(
    #         os.path.join(self.path, 'pfield_inv_for_instaseis.xml')))
    #     with ImageComparison(self.path_images, "radialtransferfunc.png") as ic:
    #         arr.plot_radial_transfer_function(0, 0.6, 0.05, [0.2])
    #         plt.savefig(ic.name)
    #
    # def test_array_plotting(self):
    #     """
    #     Tests the plotting of arrays.
    #     """
    #     arr = SeismicArray('pfield', inventory=read_inventory(
    #         os.path.join(self.path, 'pfield_inv_for_instaseis.xml')))
    #     with ImageComparison(self.path_images, "seismic_array_map.png") as ic:
    #         arr.plot()
    #         plt.savefig(ic.name)


def suite():
    return unittest.makeSuite(SeismicArrayTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')