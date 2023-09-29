import pandas as pd
import numpy as np
import pytest

from earthquakes.tools import get_haversine_distance


@pytest.fixture
def multiple_coordinates():
    """
    Fixture that generates multiple coordinates for testing purposes.

    Returns:
        Tuple: A tuple containing the following:
            - latitude_column (pd.Series): A series containing latitude values.
            - longitude_column (pd.Series): A series containing longitude values.
            - latitude (float): The latitude value to test.
            - longitude (float): The longitude value to test.
            - expected_distances (pd.Series): A series containing expected distances.
    """
    latitude_column = pd.Series([52.520008, 48.8566, 37.7749])
    longitude_column = pd.Series([13.404954, 2.3522, -122.4194])
    latitude = 52.520008
    longitude = 13.404954
    expected_distances = pd.Series([0, 878.4228495, 9115.211289])
    return latitude_column, longitude_column, latitude, longitude, expected_distances


@pytest.fixture
def single_coordinate(multiple_coordinates):
    """
    Generate a single coordinate from multiple coordinates.

    Args:
        multiple_coordinates (tuple): A tuple containing the latitude column, longitude column, latitude, longitude,
            and expected distances.

    Returns:
        tuple: A tuple containing the latitude column, longitude column, latitude, longitude, and expected distances.
    """
    (
        latitude_column,
        longitude_column,
        latitude,
        longitude,
        expected_distances,
    ) = multiple_coordinates

    return (
        pd.Series(latitude_column.iloc[0]),
        pd.Series(longitude_column.iloc[0]),
        latitude,
        longitude,
        pd.Series(expected_distances.iloc[0]),
    )


@pytest.fixture
def empty_input(multiple_coordinates):
    """
    Generates a fixture for empty input.

    Parameters:
    - multiple_coordinates: A tuple of multiple coordinates in the format (_, _, latitude, longitude, _).

    Returns:
    - An empty pandas Series for the latitude.
    - An empty pandas Series for the longitude.
    - The latitude from the multiple coordinates.
    - The longitude from the multiple coordinates.
    - An empty pandas Series.
    """
    (
        _,
        _,
        latitude,
        longitude,
        _,
    ) = multiple_coordinates

    return (
        pd.Series([]),
        pd.Series([]),
        latitude,
        longitude,
        pd.Series([]),
    )


@pytest.fixture
def large_input(single_coordinate):
    """
    Generates a large input dataset based on a single coordinate.

    Args:
        single_coordinate: A tuple containing the latitude column, longitude column,
            latitude, longitude, and expected distances.

    Returns:
        A tuple containing the latitude column repeated DATA_SIZE times, the longitude
        column repeated DATA_SIZE times, the latitude, the longitude, and the expected
        distances repeated DATA_SIZE times.
    """
    DATA_SIZE = 1000
    (
        latitude_column,
        longitude_column,
        latitude,
        longitude,
        expected_distances,
    ) = single_coordinate

    return (
        latitude_column.repeat(DATA_SIZE),
        longitude_column.repeat(DATA_SIZE),
        latitude,
        longitude,
        expected_distances.repeat(DATA_SIZE),
    )


def test_get_haversine_distance_single(single_coordinate):
    """
    Test function to verify that the calculated the haversine distance between a single coordinate and a set of
    coordinates is close to the expected distance.

    Parameters:
    - single_coordinate: A tuple containing the latitude column, longitude column, latitude, longitude, and expected distance.

    Returns:
    - None

    Raises:
    - AssertionError: If the calculated distance is not close to the expected distance.
    """
    (
        latitude_column,
        longitude_column,
        latitude,
        longitude,
        expected_distances,
    ) = single_coordinate
    distance = get_haversine_distance(
        latitude_column, longitude_column, latitude, longitude
    )
    assert np.isclose(distance, expected_distances)


def test_get_haversine_distance_multiple(multiple_coordinates):
    """
    Test function to verify that the calculated the haversine distances between a single coordinate and a set of
    coordinates is close to the expected distances.

    Args:
        multiple_coordinates (tuple): A tuple containing the latitude column, longitude column,
                                      latitude, longitude, and expected distances.

    Returns:
        None
    """
    (
        latitude_column,
        longitude_column,
        latitude,
        longitude,
        expected_distances,
    ) = multiple_coordinates

    haversine_distances = get_haversine_distance(
        latitude_column, longitude_column, latitude, longitude
    )

    assert np.allclose(
        haversine_distances,
        expected_distances,
    )


def test_get_haversine_distance_empty(empty_input):
    """
    Test is get haversine distance function returns an empty series

    Parameters:
        empty_input (tuple): A tuple containing the latitude column, longitude column,
        latitude, longitude, and expected distances.

    Returns:
        None

    Raises:
        AssertionError: If the calculated distance is not empty.
    """
    latitude_column, longitude_column, latitude, longitude, _ = empty_input
    distance = get_haversine_distance(
        latitude_column, longitude_column, latitude, longitude
    )
    assert distance.empty


def test_get_haversine_distance_large(large_input):
    """
    Test function to verify that the calculated the haversine distances between a single coordinate and a set of
    coordinates is close to the expected distances.
    The input is a large dataset of 1000 coordinates.
    """
    (
        latitude_column,
        longitude_column,
        latitude,
        longitude,
        expected_distances,
    ) = large_input

    distances = get_haversine_distance(
        latitude_column, longitude_column, latitude, longitude
    )

    assert np.allclose(distances, expected_distances)
