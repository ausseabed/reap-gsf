import datetime
import io
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import attr
import numpy
import pandas

from .data_model import (
    Comment,
    FileRecordIndex,
    History,
    PingHeader,
    PingSpatialBounds,
    PingTimestampBounds,
    SwathBathymetrySummary,
)
from .enums import BeamSubRecordTypes, RecordTypes, WGS84Coefficients

CHECKSUM_BIT = 0x80000000
NANO_SECONDS_SF = 1e-9
MAX_RECORD_ID = 12
MAX_BEAM_SUBRECORD_ID = 30


def _not_implemented(*args):
    """Handler for records we aren't reading"""
    raise NotImplementedError


def create_datetime(seconds: int, nano_seconds: int) -> datetime.datetime:
    """
    The GSF files store time as a combination of seconds and nano
    seconds past POSIX time.
    """
    timestamp = datetime.datetime.fromtimestamp(
        seconds + NANO_SECONDS_SF * nano_seconds, tz=datetime.timezone.utc
    )
    return timestamp


def record_padding(stream: Union[io.BufferedReader, io.BytesIO]) -> numpy.ndarray:
    """
    GSF requires that all records are multiples of 4 bytes.
    Essentially reads enough bytes so the stream position for the
    record finishes at a multiple of 4 bytes.
    """
    pad = stream.read(stream.tell() % 4)
    return pad


def file_info(
    stream: Union[io.BufferedReader, io.BytesIO], file_size: Optional[int] = None
) -> List[FileRecordIndex]:
    """
    Returns a list of FileRecordIndex objects for each high level record
    type in .enums.RecordTypes.
    The indexes can then be used to quickly traverse through the file.
    """
    # we could be dealing with a gsf stored within a zipfile, or as a cloud object
    if file_size is None:
        fname = Path(stream.name)
        fsize = fname.stat().st_size
    else:
        fsize = file_size

    current_pos = stream.tell()
    stream.seek(0)

    indices: Dict[RecordTypes, List[int]] = {}
    datasize: Dict[RecordTypes, List[int]] = {}
    checksum_flag: Dict[RecordTypes, List[bool]] = {}

    for rtype in RecordTypes:
        indices[rtype] = []
        datasize[rtype] = []
        checksum_flag[rtype] = []

    while stream.tell() < fsize:
        data_size, record_id, flag = read_record_info(stream)

        indices[RecordTypes(record_id)].append(stream.tell())
        datasize[RecordTypes(record_id)].append(data_size)
        checksum_flag[RecordTypes(record_id)].append(flag)

        _ = stream.read(data_size)
        _ = record_padding(stream)

    stream.seek(current_pos)

    r_index = [
        FileRecordIndex(
            record_type=rtype,
            data_size=datasize[rtype],
            checksum_flag=checksum_flag[rtype],
            indices=indices[rtype],
        )
        for rtype in RecordTypes
    ]

    return r_index


def read_record_info(
    stream: Union[io.BufferedReader, io.BytesIO]
) -> Tuple[int, int, bool]:
    """Return the header information for the current record."""
    blob = stream.read(8)
    data_size = numpy.frombuffer(blob, ">u4", count=1)[0]
    record_identifier = numpy.frombuffer(blob[4:], ">i4", count=1)[0]
    checksum_flag = bool(record_identifier & CHECKSUM_BIT)

    return data_size, record_identifier, checksum_flag


def read_header(
    stream: Union[io.BufferedReader, io.BytesIO], data_size: int, checksum_flag: bool
) -> str:
    """Read the GSF header occuring at the start of the file."""
    blob = stream.read(data_size)
    idx = 0

    if checksum_flag:
        _ = numpy.frombuffer(blob, ">i4", count=1)[0]
        idx += 4

    # TODO; if checksum is read, is data_size - 4 ??
    data = numpy.frombuffer(blob[idx:], f"S{data_size}", count=1)[0]

    _ = record_padding(stream)

    return data


def _proc_param_parser(value: Union[str, datetime.datetime]) -> Any:
    """Convert any strings that have known types such as bools, floats."""
    if isinstance(value, datetime.datetime):  # nothing to do already parsed
        return value

    booleans = {
        "yes": True,
        "no": False,
        "true": True,
        "false": False,
    }

    if "," in value:  # dealing with an array
        array = value.split(",")
        if "." in value:  # assumption on period being a decimal point
            parsed = numpy.array(array, dtype="float").tolist()
        else:
            # could be dealing with an array of "UNKNWN" or "UNKNOWN"
            parsed = ["unknown"] * len(array)
    elif "." in value:  # assumption on period being a decimal point
        parsed = float(value)
    elif value.lower() in booleans:
        parsed = booleans[value.lower()]
    elif value.lower() in ["unknwn", "unknown"]:
        parsed = "unknown"
    else:  # most likely an integer or generic string
        try:
            parsed = int(value)
        except ValueError:
            parsed = value.lower()

    return parsed


def _standardise_proc_param_keys(key: str) -> str:
    """Convert to lowercase, replace any spaces with underscore."""
    return key.lower().replace(" ", "_")


def read_processing_parameters(
    stream: Union[io.BufferedReader, io.BytesIO], data_size: int, checksum_flag: bool
) -> Dict[str, Any]:
    """
    Read the record containing the parameters used during the data
    processing phase.
    """
    idx = 0

    # blob = stream.readline(data_size)
    blob = stream.read(data_size)

    if checksum_flag:
        _ = numpy.frombuffer(blob, ">i4", count=1)[0]
        idx += 4

    dtype = numpy.dtype(
        [
            ("time_seconds", ">i4"),
            ("time_nano_seconds", ">i4"),
            ("num_params", ">i2"),
        ]
    )
    data = numpy.frombuffer(blob[idx:], dtype, count=1)
    time_seconds = int(data["time_seconds"][0])
    time_nano_seconds = int(data["time_nano_seconds"][0])

    idx += 10

    params: Dict[str, Any] = {}
    for i in range(data["num_params"][0]):
        param_size = numpy.frombuffer(blob[idx:], ">i2", count=1)[0]
        idx += 2
        data = numpy.frombuffer(blob[idx:], f"S{param_size}", count=1)[0]
        idx += param_size

        key, value = data.decode("utf-8").strip().split("=")

        if key == "REFERENCE TIME":
            value = datetime.datetime.strptime(value, "%Y/%j %H:%M:%S").replace(
                tzinfo=datetime.timezone.utc
            )
            params["processed_datetime"] = value + datetime.timedelta(
                seconds=time_seconds, milliseconds=time_nano_seconds * 1e-6
            )
            continue  # no need to include reference_time

        params[_standardise_proc_param_keys(key)] = _proc_param_parser(value)

    _ = record_padding(stream)

    return params


def read_attitude(
    stream: Union[io.BufferedReader, io.BytesIO], data_size: int, checksum_flag: bool
) -> pandas.DataFrame:
    """Read an attitude record."""
    blob = stream.read(data_size)
    idx = 0

    if checksum_flag:
        _ = numpy.frombuffer(blob, ">i4", count=1)[0]
        idx += 4

    base_time = numpy.frombuffer(blob[idx:], ">i4", count=2)
    idx += 8

    acq_time = create_datetime(base_time[0], base_time[1])

    num_measurements = numpy.frombuffer(blob[idx:], ">i2", count=1)[0]
    idx += 2

    data: Dict[str, List[Any]] = {
        "timestamp": [],
        "pitch": [],
        "roll": [],
        "heave": [],
        "heading": [],
    }

    dtype = numpy.dtype(
        [
            ("timestamp", ">i2"),
            ("pitch", ">i2"),
            ("roll", ">i2"),
            ("heave", ">i2"),
            ("heading", ">i2"),
        ]
    )
    for _ in range(num_measurements):
        numpy_blob = numpy.frombuffer(blob[idx:], dtype, count=1)[0]
        idx += 10

        data["timestamp"].append(
            acq_time + datetime.timedelta(seconds=numpy_blob["timestamp"] / 1000)
        )
        data["pitch"].append(numpy_blob["pitch"] / 100)
        data["roll"].append(numpy_blob["roll"] / 100)
        data["heave"].append(numpy_blob["heave"] / 100)
        data["heading"].append(numpy_blob["heading"] / 100)

    _ = record_padding(stream)

    dataframe = pandas.DataFrame(data)

    # as these values are stored as 2 byte integers, most precision has already
    # been truncated. therefore convert float64's to float32's
    for column in dataframe.columns:
        if "float" in dataframe.dtypes[column].name:
            dataframe[column] = dataframe[column].values.astype("float32")

    return dataframe


def read_svp(
    stream: Union[io.BufferedReader, io.BytesIO], data_size: int, flag: bool
) -> pandas.DataFrame:
    """
    Read a sound velocity profile record.
    In the provided samples, the longitude and latitude were both zero.
    It was mentioned that the datetime could be matched (or closely matched)
    with a ping, and the lon/lat could be taken from the ping.
    """
    buffer = stream.read(data_size)
    idx = 0

    dtype = numpy.dtype(
        [
            ("obs_seconds", ">u4"),
            ("obs_nano", ">u4"),
            ("app_seconds", ">u4"),
            ("app_nano", ">u4"),
            ("lon", ">i4"),
            ("lat", ">i4"),
            ("num_points", ">u4"),
        ]
    )

    blob = numpy.frombuffer(buffer, dtype, count=1)
    num_points = blob["num_points"][0]

    idx += 28

    svp = numpy.frombuffer(buffer[idx:], ">u4", count=2 * num_points) / 100
    svp = svp.reshape((num_points, 2))

    data = {
        "longitude": blob["lon"][0] / 10_000_000,
        "latitude": blob["lat"][0] / 10_000_000,
        "depth": svp[:, 0],
        "sound_velocity": svp[:, 1],
        "observation_time": create_datetime(
            blob["obs_seconds"][0], blob["obs_nano"][0]
        ),
        "applied_time": create_datetime(blob["app_seconds"][0], blob["app_nano"][0]),
    }

    _ = record_padding(stream)

    dataframe = pandas.DataFrame(
        {
            "longitude": data["longitude"] * num_points,
            "latitude": data["latitude"] * num_points,
            "depth": data["depth"] * num_points,
            "sound_velocity": data["sound_velocity"] * num_points,
            "observation_timestamp": [data["observation_time"]] * num_points,
            "applied_timestamp": [data["applied_time"]] * num_points,
        }
    )

    return dataframe


def read_swath_bathymetry_summary(
    stream: Union[io.BufferedReader, io.BytesIO], data_size: int, flag: bool
) -> SwathBathymetrySummary:
    buffer = stream.read(data_size)

    dtype = numpy.dtype(
        [
            ("time_first_ping_seconds", ">i4"),
            ("time_first_ping_nano_seconds", ">i4"),
            ("time_last_ping_seconds", ">i4"),
            ("time_last_ping_nano_seconds", ">i4"),
            ("min_latitude", ">i4"),
            ("min_longitude", ">i4"),
            ("max_latitude", ">i4"),
            ("max_longitude", ">i4"),
            ("min_depth", ">i4"),
            ("max_depth", ">i4"),
        ]
    )

    blob = numpy.frombuffer(buffer, dtype, count=1)

    data = {
        "timestamp_first_ping": create_datetime(
            blob["time_first_ping_seconds"][0], blob["time_first_ping_nano_seconds"][0]
        ),
        "timestamp_last_ping": create_datetime(
            blob["time_last_ping_seconds"][0], blob["time_last_ping_nano_seconds"][0]
        ),
        "min_latitude": blob["min_latitude"][0] / 10_000_000,
        "min_longitude": blob["min_longitude"][0] / 10_000_000,
        "max_latitude": blob["max_latitude"][0] / 10_000_000,
        "max_longitude": blob["max_longitude"][0] / 10_000_000,
        "min_depth": blob["min_depth"][0] / 100,
        "max_depth": blob["max_depth"][0] / 100,
    }

    _ = record_padding(stream)

    time_bounds = PingTimestampBounds.from_dict(data)
    spatial_bounds = PingSpatialBounds.from_dict(data)

    return SwathBathymetrySummary(time_bounds, spatial_bounds)


def read_comment(
    stream: Union[io.BufferedReader, io.BytesIO], data_size: int, flag: bool
) -> Comment:
    """Read a comment record."""
    dtype = numpy.dtype(
        [
            ("time_comment_seconds", ">i4"),
            ("time_comment_nano_seconds", ">i4"),
            ("comment_length", ">i4"),
        ]
    )
    blob = stream.read(data_size)
    decoded = numpy.frombuffer(blob, dtype, count=1)

    timestamp = create_datetime(
        decoded["time_comment_seconds"][0], decoded["time_comment_nano_seconds"][0]
    )
    the_comment = blob[12:].decode().strip().rstrip("\x00")

    _ = record_padding(stream)

    return Comment(timestamp, the_comment)


def _correct_ping_header(data):
    data_dict = {}

    data_dict["timestamp"] = create_datetime(
        data["time_ping_seconds"][0], data["time_ping_nano_seconds"][0]
    )
    data_dict["longitude"] = float(data["longitude"][0] / 10_000_000)
    data_dict["latitude"] = float(data["latitude"][0] / 10_000_000)
    data_dict["num_beams"] = int(data["number_beams"][0])
    data_dict["center_beam"] = int(data["centre_beam"][0])
    data_dict["ping_flags"] = int(data["ping_flags"][0])
    data_dict["reserved"] = int(data["reserved"][0])
    data_dict["tide_corrector"] = float(data["tide_corrector"][0] / 100)
    data_dict["depth_corrector"] = float(data["depth_corrector"][0] / 100)
    data_dict["heading"] = float(data["heading"][0] / 100)
    data_dict["pitch"] = float(data["pitch"][0] / 100)
    data_dict["roll"] = float(data["roll"][0] / 100)
    data_dict["heave"] = float(data["heave"][0] / 100)
    data_dict["course"] = float(data["course"][0] / 100)
    data_dict["speed"] = float(data["speed"][0] / 100)
    data_dict["height"] = float(data["height"][0] / 1000)
    data_dict["separation"] = float(data["separation"][0] / 1000)
    data_dict["gps_tide_corrector"] = float(data["gps_tide_corrector"][0] / 1000)

    ping_header = PingHeader(**data_dict)

    return ping_header


def _beams_longitude_latitude(
    ping_header: PingHeader, along_track: numpy.ndarray, across_track: numpy.ndarray
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    Calculate the longitude and latitude for each beam.

    https://en.wikipedia.org/wiki/Geographic_coordinate_system
    For lonitude and latitude calculations:
        * lat_m_sf = A - B * cos(2 * lat) + C  * cos(4 * lat) - D * cos(6 * lat)
        * lon_m_sf = E * cos(lat) - F * cos(3 * lat) + G * cos(5 * lat)
    """
    # see https://math.stackexchange.com/questions/389942/why-is-it-necessary-to-use-sin-or-cos-to-determine-heading-dead-reckoning # noqa: E501
    lat_radians = math.radians(ping_header.latitude)

    coef_a = WGS84Coefficients.A.value
    coef_b = WGS84Coefficients.B.value
    coef_c = WGS84Coefficients.C.value
    coef_d = WGS84Coefficients.D.value
    coef_e = WGS84Coefficients.E.value
    coef_f = WGS84Coefficients.F.value
    coef_g = WGS84Coefficients.G.value

    lat_mtr_sf = (
        coef_a
        - coef_b * math.cos(2 * lat_radians)
        + coef_c * math.cos(4 * lat_radians)
        - coef_d * math.cos(6 * lat_radians)
    )
    lon_mtr_sf = (
        coef_e * math.cos(lat_radians)
        - coef_f * math.cos(3 * lat_radians)
        + coef_g * math.cos(5 * lat_radians)
    )

    delta_x = math.sin(math.radians(ping_header.heading))
    delta_y = math.cos(math.radians(ping_header.heading))

    lon2 = (
        ping_header.longitude
        + delta_y / lon_mtr_sf * across_track
        + delta_x / lon_mtr_sf * along_track
    )
    lat2 = (
        ping_header.latitude
        - delta_x / lat_mtr_sf * across_track
        + delta_y / lat_mtr_sf * along_track
    )

    return lon2, lat2


def _ping_dataframe(
    ping_header: PingHeader, subrecords: Dict[BeamSubRecordTypes, numpy.ndarray]
) -> pandas.DataFrame:
    """Construct the dataframe for the given ping."""
    # convert beam arrays to point cloud structure (i.e. generate coords for every beam)
    longitude, latitude = _beams_longitude_latitude(
        ping_header,
        subrecords[BeamSubRecordTypes.ALONG_TRACK],
        subrecords[BeamSubRecordTypes.ACROSS_TRACK],
    )

    dataframe = pandas.DataFrame({k.name.lower(): v for k, v in subrecords.items()})
    dataframe.insert(0, "latitude", latitude)
    dataframe.insert(0, "longitude", longitude)

    # include the header info in the dataframe as that was desired by many in the survey
    ignore = [
        "longitude",
        "latitude",
        "num_beams",
        "reserved",
        "center_beam",
    ]
    for key, value in attr.asdict(ping_header).items():
        if key in ignore:
            continue
        if key == "timestamp":
            value = value.replace(tzinfo=None)
        dataframe[key] = value

    # most perpendicular beam
    dataframe["centre_beam"] = False
    if ping_header.ping_flags == 0:
        query = dataframe.beam_flags == 0
        subset = dataframe[query]
        if subset.shape[0]:
            # require suitable beams from this ping to determine the centre beam
            idx = subset.across_track.abs().idxmin()
            dataframe.loc[idx, "centre_beam"] = True

    # beam number
    dataframe["beam_number"] = numpy.arange(ping_header.num_beams).astype("uint16")

    # float32 conversion;
    # it seems all the attributes have had some level of truncation applied
    # thus losing some level of precision

    ignore = ["longitude", "latitude"]
    for column in dataframe.columns:
        if column in ignore:
            continue
        if "float" in dataframe.dtypes[column].name:
            dataframe[column] = dataframe[column].values.astype("float32")

    return dataframe


def _ping_scale_factors(
    num_factors: int, buffer: str, idx: int
) -> Tuple[Dict[BeamSubRecordTypes, numpy.ndarray], int]:
    """Small util for populating the ping scale factors."""
    scale_factors: Dict[BeamSubRecordTypes, numpy.ndarray] = {}

    for i in range(num_factors):
        blob = numpy.frombuffer(buffer[idx:], ">i4", count=3)
        beam_subid = (blob[0] & 0xFF000000) >> 24
        _ = (blob & 0x00FF0000) >> 16  # compression flag

        scale_factors[BeamSubRecordTypes(beam_subid)] = blob[1:]
        idx = idx + 12

    return scale_factors, idx


def _ping_beam_subrecord(
    ping_header: PingHeader,
    buffer: str,
    scale_factors: Dict[BeamSubRecordTypes, numpy.ndarray],
    subrecord_size: int,
    subrecord_id: int,
    idx: int,
) -> Tuple[numpy.ndarray, int, int, int]:
    """Small util for reading and converting a ping beam subrecord."""
    size = subrecord_size // ping_header.num_beams
    sub_rec_type = BeamSubRecordTypes(subrecord_id)
    dtype = f"{sub_rec_type.dtype_mapper}{size}"
    sub_rec_blob = numpy.frombuffer(buffer[idx:], dtype, count=ping_header.num_beams)

    idx = idx + size * ping_header.num_beams

    scale = scale_factors[sub_rec_type][0]
    offset = scale_factors[sub_rec_type][1]

    data = sub_rec_blob / scale - offset

    subrecord_hdr = numpy.frombuffer(buffer[idx:], ">i4", count=1)[0]
    idx = idx + 4

    subrecord_id = (subrecord_hdr & 0xFF000000) >> 24
    subrecord_size = subrecord_hdr & 0x00FFFFFF

    return data, subrecord_id, subrecord_size, idx


def read_bathymetry_ping(
    stream, data_size, flag, scale_factors=None
) -> Tuple[PingHeader, Dict[BeamSubRecordTypes, numpy.ndarray], pandas.DataFrame]:
    """Read and digest a bathymetry ping record."""
    idx = 0
    blob = stream.read(data_size)

    dtype = numpy.dtype(
        [
            ("time_ping_seconds", ">i4"),
            ("time_ping_nano_seconds", ">i4"),
            ("longitude", ">i4"),
            ("latitude", ">i4"),
            ("number_beams", ">i2"),
            ("centre_beam", ">i2"),
            ("ping_flags", ">i2"),
            ("reserved", ">i2"),
            ("tide_corrector", ">i2"),
            ("depth_corrector", ">i4"),
            ("heading", ">u2"),
            ("pitch", ">i2"),
            ("roll", ">i2"),
            ("heave", ">i2"),
            ("course", ">u2"),
            ("speed", ">u2"),
            ("height", ">i4"),
            ("separation", ">i4"),
            ("gps_tide_corrector", ">i4"),
        ]
    )

    ping_header = _correct_ping_header(numpy.frombuffer(blob, dtype=dtype, count=1))

    idx += 56  # includes 2 bytes of spare space

    # first subrecord
    subrecord_hdr = numpy.frombuffer(blob[idx:], ">i4", count=1)[0]
    subrecord_id = (subrecord_hdr & 0xFF000000) >> 24
    subrecord_size = subrecord_hdr & 0x00FFFFFF

    idx += 4

    if subrecord_id == 100:
        # scale factor subrecord
        num_factors = numpy.frombuffer(blob[idx:], ">i4", count=1)[0]
        idx += 4

        # if we have input sf's return new ones
        # some pings don't store a scale factor record and rely on
        # ones read from a previous ping
        scale_factors, idx = _ping_scale_factors(num_factors, blob, idx)
    else:
        if scale_factors is None:
            # can't really do anything sane
            # could return the unscaled data, but that's not the point here
            raise Exception("Record has no scale factors")

        # roll back the index by 4 bytes
        idx -= 4

    subrecord_hdr = numpy.frombuffer(blob[idx:], ">i4", count=1)[0]
    idx += 4

    subrecord_id = (subrecord_hdr & 0xFF000000) >> 24
    subrecord_size = subrecord_hdr & 0x00FFFFFF

    # beam array subrecords
    subrecords = {}
    while subrecord_id <= MAX_BEAM_SUBRECORD_ID:
        data, new_subrecord_id, subrecord_size, idx = _ping_beam_subrecord(
            ping_header, blob, scale_factors, subrecord_size, subrecord_id, idx
        )
        subrecords[BeamSubRecordTypes(subrecord_id)] = data
        subrecord_id = new_subrecord_id

    dataframe = _ping_dataframe(ping_header, subrecords)

    # SKIPPING:
    #     * sensor specific sub records
    #     * intensity series
    # print(idx)

    return ping_header, scale_factors, dataframe


def read_history(
    stream: Union[io.BufferedReader, io.BytesIO], data_size: int, flag: bool
):
    """Read a history record."""
    blob = stream.read(data_size)
    idx = 0

    time = numpy.frombuffer(blob, ">i4", count=2)
    timestamp = create_datetime(time[0], time[1])
    idx += 8

    size = numpy.frombuffer(blob[idx:], ">i2", count=1)[0]
    idx += 2

    end_idx = idx + size + 1
    machine_name = blob[idx:end_idx].decode().strip().rstrip("\x00")
    idx += size

    size = numpy.frombuffer(blob[idx:], ">i2", count=1)[0]
    idx += 2

    end_idx = idx + size + 1
    operator_name = blob[idx:end_idx].decode().strip().rstrip("\x00")
    idx += size

    size = numpy.frombuffer(blob[idx:], ">i2", count=1)[0]
    idx += 2

    end_idx = idx + size + 1
    command = blob[idx:end_idx].decode().strip().rstrip("\x00")
    idx += size

    size = numpy.frombuffer(blob[idx:], ">i2", count=1)[0]
    idx += 2

    end_idx = idx + size + 1
    comment = blob[idx:end_idx].decode().strip().rstrip("\x00")

    history = History(timestamp, machine_name, operator_name, command, comment)

    return history
