import io
import urllib

import boto3
import pytest

from reap_gsf import reap

URI = "s3://ausseabed-pl019-baseline-data/original-samples/0364_BeagleMarinePark/L2/GSF/MV_Bluefin_Single_2040C_300kHz/2018-169/0072_20180618_035231_BlueFin.gsf"  # noqa: E501


@pytest.fixture(scope="module")
def retrieve_stream():
    """
    Not testing the creation of the stream object at this point.
    But for testing, we also need to keep the download to occur only
    once.
    """
    dev_resource = boto3.resource("s3")
    uri = urllib.parse.urlparse(URI)
    obj = dev_resource.Object(bucket_name=uri.netloc, key=uri.path[1:])
    stream = io.BytesIO(obj.get()["Body"].read())

    return stream, obj.content_length


@pytest.fixture(scope="module")
def file_info(retrieve_stream):
    """Retrieve the GSF file info."""
    stream, file_size = retrieve_stream
    return reap.file_info(stream, file_size)


def test_file_info(retrieve_stream):
    """Keeping it simple for the time being."""
    stream, file_size = retrieve_stream
    try:
        reap.file_info(stream, file_size)
    except Exception as err:
        raise AssertionError(f"reap.file_info failed {err}")


def test_header_count(file_info):
    """Test the header record count."""
    assert file_info[0].record_count == 1


def test_swath_bathymetry_ping_count(file_info):
    """Test the header record count."""
    assert file_info[1].record_count == 954


def test_sound_velocity_profile_count(file_info):
    """Test the header record count."""
    assert file_info[2].record_count == 1


def test_processing_parameters_count(file_info):
    """Test the header record count."""
    assert file_info[3].record_count == 1


def test_sensor_parameters_count(file_info):
    """Test the header record count."""
    assert file_info[4].record_count == 0


def test_comment_count(file_info):
    """Test the header record count."""
    assert file_info[5].record_count == 3


def test_history_count(file_info):
    """Test the header record count."""
    assert file_info[6].record_count == 1


def test_navigation_error_count(file_info):
    """Test the header record count."""
    assert file_info[7].record_count == 0


def test_swath_bathymetry_summary_count(file_info):
    """Test the header record count."""
    assert file_info[8].record_count == 1


def test_single_beam_ping_count(file_info):
    """Test the header record count."""
    assert file_info[9].record_count == 0


def test_hv_navigation_error_count(file_info):
    """Test the header record count."""
    assert file_info[10].record_count == 0


def test_attitude_count(file_info):
    """Test the header record count."""
    assert file_info[11].record_count == 535
