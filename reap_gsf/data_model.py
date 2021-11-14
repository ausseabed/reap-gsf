import datetime  # type: ignore
from typing import List, Union

import attr
import pandas  # type: ignore

from .enums import RecordTypes


@attr.s()
class Record:
    """Instance of a GSF high level record as referenced in RecordTypes."""

    record_type: RecordTypes = attr.ib()
    data_size: int = attr.ib()
    checksum_flag: bool = attr.ib()
    index: int = attr.ib()
    record_index: int = attr.ib()

    def read(self, stream, *args):
        """Read the data associated with this record."""
        stream.seek(self.index)
        data = self.record_type.func_mapper(
            stream, self.data_size, self.checksum_flag, *args
        )
        return data


@attr.s()
class FileRecordIndex:

    record_type: RecordTypes = attr.ib()
    record_count: int = attr.ib(init=False)
    data_size: List[int] = attr.ib(repr=False)
    checksum_flag: List[bool] = attr.ib(repr=False)
    indices: List[int] = attr.ib(repr=False)

    def __attrs_post_init__(self):
        self.record_count = len(self.indices)

    def record(self, index):
        result = Record(
            record_type=self.record_type,
            data_size=self.data_size[index],
            checksum_flag=self.checksum_flag[index],
            index=self.indices[index],
            record_index=index,
        )
        return result


@attr.s()
class SwathBathymetryPing:
    """
    Data model class for the SwathBathymetryPing sub-records contained
    within a GSF file.
    Essentially all records are combined into a tabular form as a
    pandas.DataFrame construct.
    """

    file_record: FileRecordIndex = attr.ib()
    ping_dataframe: pandas.DataFrame = attr.ib()
    # sensor_dataframe: pandas.DataFrame = attr.ib()

    @classmethod
    def from_records(cls, file_record, stream):
        """Constructor for SwathBathymetryPing"""
        rec = file_record.record(0)
        ping_header, scale_factors, ping_dataframe = rec.read(stream)

        for i in range(1, file_record.record_count):
            rec = file_record.record(i)

            # some pings don't have scale factors and rely on a previous ping
            ping_header, scale_factors, df = rec.read(stream, scale_factors)

            # this isn't the most efficient way
            # we could pre-allocate the entire array, but i can't be certain that
            # each ping has the same number of beams
            ping_dataframe = ping_dataframe.append(df, ignore_index=True)

        ping_dataframe.reset_index(drop=True, inplace=True)

        return cls(file_record, ping_dataframe)


@attr.s()
class PingHeader:
    """
    Data model class for a swath bathymetry ping header record.
    The ping header comes before the ping sub-records that contain
    the beam array for the current ping.
    """

    timestamp: datetime.datetime = attr.ib()
    longitude: float = attr.ib()
    latitude: float = attr.ib()
    num_beams: int = attr.ib()
    center_beam: int = attr.ib()
    ping_flags: int = attr.ib()
    reserved: int = attr.ib()
    tide_corrector: int = attr.ib()
    depth_corrector: int = attr.ib()
    heading: float = attr.ib()
    pitch: float = attr.ib()
    roll: float = attr.ib()
    heave: int = attr.ib()
    course: float = attr.ib()
    speed: float = attr.ib()
    height: int = attr.ib()
    separation: int = attr.ib()
    gps_tide_corrector: int = attr.ib()


@attr.s()
class Comment:
    """
    Container for a single comment record.
    """

    timestamp: datetime.datetime = attr.ib()
    comment: str = attr.ib()


@attr.s()
class Comments:
    """
    Construct to read and hold all comments within a GSF file.
    """

    file_record: FileRecordIndex = attr.ib()
    comments: Union[List[Comment], None] = attr.ib(default=None)

    @classmethod
    def from_records(cls, file_record, stream):
        """Constructor for all the comments in the GSF file."""
        comments = []
        for i in range(file_record.record_count):
            record = file_record.record(i)
            data = record.read(stream)
            comments.append(data)

        return cls(file_record, comments)


@attr.s
class PingSpatialBounds:

    min_x: Union[float, None] = attr.ib(default=None)
    min_y: Union[float, None] = attr.ib(default=None)
    min_z: Union[float, None] = attr.ib(default=None)
    max_x: Union[float, None] = attr.ib(default=None)
    max_y: Union[float, None] = attr.ib(default=None)
    max_z: Union[float, None] = attr.ib(default=None)

    @classmethod
    def from_dict(cls, bounds):
        """Constructor for PingSpatialBounds."""
        min_x = bounds.get("min_longitude", None)
        min_y = bounds.get("min_latitude", None)
        min_z = bounds.get("min_depth", None)
        max_x = bounds.get("max_longitude", None)
        max_y = bounds.get("max_latitude", None)
        max_z = bounds.get("max_depth", None)

        return cls(min_x, min_y, min_z, max_x, max_y, max_z)


@attr.s
class PingTimestampBounds:
    first_ping: Union[datetime.datetime, None] = attr.ib(default=None)
    last_ping: Union[datetime.datetime, None] = attr.ib(default=None)

    @classmethod
    def from_dict(cls, bounds):
        """Constructor for PingTimestampBounds."""
        first_ping = bounds.get("timestamp_first_ping", None)
        last_ping = bounds.get("timestamp_first_ping", None)

        return cls(first_ping, last_ping)


@attr.s()
class SwathBathymetrySummary:
    """
    Container for the swath bathymetry summary record.
    """

    timestamp_bounds: PingTimestampBounds = attr.ib()
    spatial_bounds: PingSpatialBounds = attr.ib()


@attr.s()
class Attitude:
    """Data model to combine all attitude records."""

    file_record: FileRecordIndex = attr.ib()
    attitude_dataframe: pandas.DataFrame = attr.ib()

    @classmethod
    def from_records(cls, file_record, stream):
        """Constructor for Attitude."""
        rec = file_record.record(0)
        dataframe = rec.read(stream)

        for i in range(1, file_record.record_count):
            rec = file_record.record(i)
            try:
                dataframe = dataframe.append(rec.read(stream), ignore_index=True)
            except ValueError as err:
                msg = f"record: {rec}, iteration: {i}"
                print(msg, err)
                raise Exception

        dataframe.reset_index(drop=True, inplace=True)

        return cls(file_record, dataframe)


@attr.s()
class History:
    """Container for a history record."""

    processing_timestamp: datetime.datetime = attr.ib()
    machine_name: str = attr.ib()
    operator_name: str = attr.ib()
    command: str = attr.ib()
    comment: str = attr.ib()
