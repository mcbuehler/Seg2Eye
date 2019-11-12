from collections import OrderedDict
import logging
import os
import socket
import time
import traceback

import gspread
import numpy as np
from oauth2client.service_account import ServiceAccountCredentials

from core import DefaultConfig

config = DefaultConfig()
logger = logging.getLogger(__name__)


class GoogleSheetLogger:
    """Log selected outputs to a predefined Google Sheet.

    Many thanks to Emre for the idea and the initial code!
    """

    first_column_name = 'Identifier'

    def __init__(self, model):
        self.__model = model
        to_write = OrderedDict()
        to_write['Identifier'] = self.__model.identifier

        # Write config parameters
        config_kv = config.get_all_key_values()
        config_kv = dict([
            (k, v) for k, v in config_kv.items()
            if not k.startswith('datasrc_')
            and not k.startswith('gsheet_')
        ])
        for k in sorted(list(config_kv.keys())):
            to_write[k] = config_kv[k]

        # Get hostname
        to_write['hostname'] = socket.getfqdn()
        to_write['Start Time'] = time.strftime('%Y/%m/%d %H:%M:%S')

        # Get LSF job ID if exists
        if 'LSB_JOBID' in os.environ:
            to_write['LSF Job ID'] = os.environ['LSB_JOBID']

        # Write experiment information to create row for future logging
        try:
            self.ready = True
            self.update_or_append_row(to_write)
        except Exception:
            self.ready = False
            traceback.print_exc()
            return

    def update_or_append_row(self, values):
        assert isinstance(values, dict)

        if not self.ready:  # Silently skip if init failed
            return

        # Add some missing info automatically
        if 'Identifier' not in values:
            values['Identifier'] = self.__model.identifier
        values['Last Updated'] = time.strftime('%Y/%m/%d %H:%M:%S')

        # Authenticate
        try:
            credentials = ServiceAccountCredentials.from_json_keyfile_name(
                filename=config.gsheet_secrets_json_file,
                scopes=[
                    'https://www.googleapis.com/auth/spreadsheets',
                ],
            )
            client = gspread.authorize(credentials)
        except:  # noqa
            logger.debug('Could not authenticate with Drive API.')
            traceback.print_exc()
            return

        # Decide on sheet name to select
        sheet_name = self.__model.__class__.__name__

        # Find a workbook by name.
        workbook = client.open_by_key(config.gsheet_workbook_key)
        try:
            sheet = workbook.worksheet(sheet_name)
        except:  # noqa
            sheet = workbook.add_worksheet(title=sheet_name, rows=1000, cols=20)
        current_values = sheet.get_all_values()
        if len(current_values) == 0:
            sheet.update_cell(1, 1, self.first_column_name)
            header = [self.first_column_name]
        else:
            header = current_values[0]

        identifier = values[self.first_column_name]

        # Construct new row
        is_header_changed = False
        new_row = [None] * len(header)
        for key, value in values.items():
            if key not in header:
                header.append(key)
                new_row.append(None)
                is_header_changed = True
            index = header.index(key)
            if isinstance(value, np.generic):
                if np.isnan(value):
                    new_row[index] = 'NaN'
                elif np.isinf(value):
                    new_row[index] = 'Inf'
                else:
                    new_row[index] = np.asscalar(value)
            elif isinstance(value, np.ndarray) and value.ndim == 0:
                new_row[index] = value.item()
            elif hasattr(value, '__len__') and len(value) > 0:
                new_row[index] = str(value)
            else:
                new_row[index] = value

        # Update header as necessary
        cells_to_update = []
        if is_header_changed:
            cells_to_update += [
                gspread.models.Cell(1, col+1, value)
                for col, value in enumerate(header)
            ]

        # Either update an existing row or append new row
        try:
            row_index = [r[0] for r in current_values].index(identifier)
            cells_to_update += [
                gspread.models.Cell(row_index+1, col_index+1, value=value)
                for col_index, value in enumerate(new_row)
                if value is not None  # Don't remove existing values
            ]
        except:  # noqa
            sheet.append_row(new_row)

        # Run all necessary update operations
        if len(cells_to_update) > 0:
            sheet.update_cells(cells_to_update)
