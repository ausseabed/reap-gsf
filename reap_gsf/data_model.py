import datetime  # type: ignore
from typing import List, Union

import attr
import numpy
import pandas  # type: ignore

from .enums import RecordTypes


def _dependent_pings(stream, file_record, idx=slice(None)):
    """
    Return a list of dependent pings. This is to aid which pings require scale
    factors from a previous ping.
    """
    results = []
    for i in range(file_record.record_count):
        record = file_record.record(i)
        stream.seek(record.index)
        buffer = stream.read(60)
        subhdr = numpy.frombuffer(buffer[56:], ">i4", count=1)[0]
        subid = (subhdr & 0xFF000000) >> 24
        if subid == 100:
            dep_id = i
            dep = False
        else:
            dep = True
        results.append((dep, dep_id))

    return results[idx]


def _total_ping_beam_count(stream, file_record, idx=slice(None)):
    """
    Return a the total ping beam count.
    The basis for this is that (despite what we were told), the beam count
    can differ between pings. So in order to read slices and insert slices
    into a pre-allocated array, we now need to know the beam count for every
    ping within the slice.
    """
    results = []
    for i in range(file_record.record_count):
        record = file_record.record(i)
        ping_hdr = record.read(stream, None, True)  # read header only
        results.append(ping_hdr.num_beams)

    return numpy.sum(results[idx])


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
    def from_records(cls, file_record, stream, idx=slice(None)):
        """Constructor for SwathBathymetryPing. Not supporting idx.step > 1"""
        # TODO testing
        # retrieve the full ping, and a subset.
        # result = full_df[idx.start*nbeams:idx.end*nbeams].reset_index(drop=True) - subs
        # (result.sum() == 0).all() (timestamp won't work, but should be 0 days 00:00:00)
        # ~(result.all()).all() should do the jo
        record_index = list(range(file_record.record_count))
        record_ids = record_index[idx]

        # record dependencies (required for scale factors)
        # only need to resolve the first record as subsequent records are
        # provided with scale_factors
        dependent_pings = _dependent_pings(stream, file_record, idx)

        # get the first record of interest
        if dependent_pings[0][0]:
            rec = file_record.record(dependent_pings[0][1])
            ping_header, scale_factors, df = rec.read(stream)

            rec = file_record.record(record_ids[0])
            ping_header, scale_factors, df = rec.read(stream, scale_factors)
        else:
            rec = file_record.record(record_ids[0])
            ping_header, scale_factors, df = rec.read(stream)

        # allocating the full dataframe upfront is an attempt to reduce the
        # memory footprint. the append method allocates a whole new copy
        # nrows = file_record.record_count * ping_header.num_beams
        # nrows = len(record_ids) * ping_header.num_beams
        nrows = _total_ping_beam_count(stream, file_record, idx)
        ping_dataframe = pandas.DataFrame(
            {
                column: numpy.empty((nrows), dtype=df[column].dtype)
                for column in df.columns
            }
        )

        slices = [
            slice(start, start + ping_header.num_beams)
            for start in numpy.arange(0, nrows, ping_header.num_beams)
        ]
        ping_dataframe[slices[0]] = df

        for i, rec_id in enumerate(record_ids[1:]):
            rec = file_record.record(rec_id)

            # some pings don't have scale factors and rely on a previous ping
            ping_header, scale_factors, df = rec.read(stream, scale_factors)

            ping_dataframe[slices[i + 1]] = df

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
        last_ping = bounds.get("timestamp_last_ping", None)

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
