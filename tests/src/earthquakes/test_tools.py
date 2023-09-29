import pandas as pd
import numpy as np
import pytest

from earthquakes.tools import get_haversine_distance


@pytest.fixture
def multiple_coordinates():
    latitude_column = pd.Series([52.520008, 48.8566, 37.7749])
    longitude_column = pd.Series([13.404954, 2.3522, -122.4194])
    latitude = 52.520008
    longitude = 13.404954
    expected_distances = pd.Series([0, 878.4228495, 9115.211289])
    return latitude_column, longitude_column, latitude, longitude, expected_distances


@pytest.fixture
def single_coordinate(multiple_coordinates):
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
    latitude_column, longitude_column, latitude, longitude, _ = empty_input
    distance = get_haversine_distance(
        latitude_column, longitude_column, latitude, longitude
    )
    assert distance.empty


def test_get_haversine_distance_large(large_input):
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
