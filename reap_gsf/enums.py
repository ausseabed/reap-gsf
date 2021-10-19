from enum import Enum

# from .reap import read_header, read_bathymetry_ping, read_svp, read_processing_parameters, _not_implemented, read_comment, read_swath_bathymetry_summary, read_attitude
from reap_gsf import reap


class WGS84Coefficients(Enum):
    """
    https://en.wikipedia.org/wiki/Geographic_coordinate_system
    For lonitude and latitude calculations:
        * lat_m_sf = A - B * cos(2 * lat) + C  * cos(4 * lat) - D * cos(6 * lat)
        * lon_m_sf = E * cos(lat) - F * cos(3 * lat) + G * cos(5 * lat)
    """

    A = 111132.92
    B = 559.82
    C = 1.175
    D = 0.0023
    E = 111412.84
    F = 93.5
    G = 0.118


class RecordTypes(Enum):
    """The various record type contained within the GSF file."""

    GSF_HEADER = 1
    GSF_SWATH_BATHYMETRY_PING = 2
    GSF_SOUND_VELOCITY_PROFILE = 3
    GSF_PROCESSING_PARAMETERS = 4
    GSF_SENSOR_PARAMETERS = 5
    GSF_COMMENT = 6
    GSF_HISTORY = 7
    GSF_NAVIGATION_ERROR = 8
    GSF_SWATH_BATHY_SUMMARY = 9
    GSF_SINGLE_BEAM_PING = 10
    GSF_HV_NAVIGATION_ERROR = 11
    GSF_ATTITUDE = 12

    @property
    def func_mapper(self):
        func_map = {
            RecordTypes.GSF_HEADER: reap.read_header,
            RecordTypes.GSF_SWATH_BATHYMETRY_PING: reap.read_bathymetry_ping,
            RecordTypes.GSF_SOUND_VELOCITY_PROFILE: reap.read_svp,
            RecordTypes.GSF_PROCESSING_PARAMETERS: reap.read_processing_parameters,
            RecordTypes.GSF_SENSOR_PARAMETERS: reap._not_implemented,
            RecordTypes.GSF_COMMENT: reap.read_comment,
            RecordTypes.GSF_HISTORY: reap.read_history,
            RecordTypes.GSF_NAVIGATION_ERROR: reap._not_implemented,
            RecordTypes.GSF_SWATH_BATHY_SUMMARY: reap.read_swath_bathymetry_summary,
            RecordTypes.GSF_SINGLE_BEAM_PING: reap._not_implemented,
            RecordTypes.GSF_HV_NAVIGATION_ERROR: reap._not_implemented,
            RecordTypes.GSF_ATTITUDE: reap.read_attitude,
        }
        return func_map.get(self)


class BeamSubRecordTypes(Enum):
    """The Swath Bathymetry Ping subrecord ID's."""

    DEPTH = 1
    ACROSS_TRACK = 2
    ALONG_TRACK = 3
    TRAVEL_TIME = 4
    BEAM_ANGLE = 5
    MEAN_CAL_AMPLITUDE = 6
    MEAN_REL_AMPLITUDE = 7
    ECHO_WIDTH = 8
    QUALITY_FACTOR = 9
    RECEIVE_HEAVE = 10
    DEPTH_ERROR = 11  # obselete
    ACROSS_TRACK_ERROR = 12  # obselete
    ALONG_TRACK_ERROR = 13  # obselete
    NOMINAL_DEPTH = 14
    QUALITY_FLAGS = 15
    BEAM_FLAGS = 16
    SIGNAL_TO_NOISE = 17
    BEAM_ANGLE_FORWARD = 18
    VERTICAL_ERROR = 19
    HORIZONTAL_ERROR = 20
    INTENSITY_SERIES = 21
    SECTOR_NUMBER = 22
    DETECTION_INFO = 23
    INCIDENT_BEAM_ADJ = 24
    SYSTEM_CLEANING = 25
    DOPPLER_CORRECTION = 26
    SONAR_VERT_UNCERNTAINTY = 27
    SONAR_HORZ_UNCERTAINTY = 28
    DETECTION_WINDOW = 29
    MEAN_ABS_COEF = 30

    @property
    def dtype_mapper(self):
        dtype_map = {
            BeamSubRecordTypes.DEPTH: ">u",
            BeamSubRecordTypes.ACROSS_TRACK: ">i",
            BeamSubRecordTypes.ALONG_TRACK: ">i",
            BeamSubRecordTypes.TRAVEL_TIME: ">u",
            BeamSubRecordTypes.BEAM_ANGLE: ">i",
            BeamSubRecordTypes.MEAN_CAL_AMPLITUDE: ">i",
            BeamSubRecordTypes.MEAN_REL_AMPLITUDE: ">i",
            BeamSubRecordTypes.ECHO_WIDTH: ">u",
            BeamSubRecordTypes.QUALITY_FACTOR: ">u",
            BeamSubRecordTypes.RECEIVE_HEAVE: ">i",
            BeamSubRecordTypes.DEPTH_ERROR: ">u",
            BeamSubRecordTypes.ACROSS_TRACK_ERROR: ">u",
            BeamSubRecordTypes.ALONG_TRACK_ERROR: ">u",
            BeamSubRecordTypes.NOMINAL_DEPTH: ">u",
            BeamSubRecordTypes.QUALITY_FLAGS: ">u",
            BeamSubRecordTypes.BEAM_FLAGS: ">u",
            BeamSubRecordTypes.SIGNAL_TO_NOISE: ">i",
            BeamSubRecordTypes.BEAM_ANGLE_FORWARD: ">u",
            BeamSubRecordTypes.VERTICAL_ERROR: ">u",
            BeamSubRecordTypes.HORIZONTAL_ERROR: ">u",
            BeamSubRecordTypes.INTENSITY_SERIES: ">i",  # not a single type
            BeamSubRecordTypes.SECTOR_NUMBER: ">i",
            BeamSubRecordTypes.DETECTION_INFO: ">i",
            BeamSubRecordTypes.INCIDENT_BEAM_ADJ: ">i",
            BeamSubRecordTypes.SYSTEM_CLEANING: ">i",
            BeamSubRecordTypes.DOPPLER_CORRECTION: ">i",
            BeamSubRecordTypes.SONAR_VERT_UNCERNTAINTY: ">i",  # dtype not defined in 3.09 pdf
            BeamSubRecordTypes.SONAR_HORZ_UNCERTAINTY: ">i",  # dtype and record not defined in 3.09 pdf
            BeamSubRecordTypes.DETECTION_WINDOW: ">i",  # dtype and record not defined in 3.09 pdf
            BeamSubRecordTypes.MEAN_ABS_COEF: ">i",  # dtype and record not defined in 3.09 pdf
        }
        return dtype_map.get(self)


class SensorSpecific(Enum):
    EM2040 = 149
