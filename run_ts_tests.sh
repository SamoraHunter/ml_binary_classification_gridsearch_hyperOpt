#!/bin/bash
# Script to run only the time series unit tests
pytest tests/test_time_series_models.py -m time_series
