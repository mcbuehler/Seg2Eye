from collections import OrderedDict
import logging
import math
import os
import socket
import time
import traceback

import gspread
import numpy as np
from oauth2client.service_account import ServiceAccountCredentials


class GoogleSheetLogger:
    """Log selected outputs to a predefined Google Sheet.

    Many thanks to Emre for the idea and the initial code!
    """
    def __init__(self, opt):
        # self.__model = model
        self.opt = opt
        to_write = OrderedDict()
        to_write['Name'] = self.opt.name

        # Write config parameters
        config_kv = OrderedDict((k, v) for k, v in sorted(vars(opt).items()))

        for k, v in config_kv.items():
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

    def _get_worksheet(self):
        # Authenticate
        try:
            credentials = ServiceAccountCredentials.from_json_keyfile_name(
                filename=self.opt.gsheet_secrets_json_file,
                scopes=[
                    'https://www.googleapis.com/auth/spreadsheets',
                ],
            )
            client = gspread.authorize(credentials)
        except:  # noqa
            print('Could not authenticate with Drive API.')
            traceback.print_exc()
            return

        # Find a workbook by name.
        workbook = client.open_by_key("1F8KydmPlO49SgRkRJdBjaa3XomLKNkz7TSqYTpWIQ2s")
        try:
            # sheet = workbook.worksheet(sheet_name)
            sheet = workbook.get_worksheet(0)
        except:  # noqa
            print("Error when loading sheet.")
            traceback.print_exc()
            return
            # sheet = workbook.add_worksheet(title=sheet_name, rows=1000, cols=20)
        return sheet

    def update_or_append_row(self, values):
        assert isinstance(values, dict)
        # having empty lists in here yields an error. To save time we add this hacky filter.
        values = OrderedDict(e for e in values.items() if e[0] != 'gpu_ids')

        if not self.ready:  # Silently skip if init failed
            return

        sheet = self._get_worksheet()

        values['Last Updated'] = time.strftime('%Y/%m/%d %H:%M:%S')
        current_values = sheet.get_all_values()
        if len(current_values) == 0:
            sheet.update_cell(1, 1, 'Name')
            header = ['Name']
        else:
            header = current_values[0]

        identifier = self.opt.name

        # Construct new row
        is_header_changed = False
        new_row = [None] * len(header)
        for key, value in values.items():
            if key not in header:
                header.append(key)
                new_row.append(None)
                is_header_changed = True
            index = header.index(key)
            new_row[index] = value
            if isinstance(value, float) or isinstance(value, int):
                if math.isnan(value):
                    new_row[index] = 'NaN'
            elif isinstance(value, np.generic):
                if np.any(np.isnan(value)):
                    new_row[index] = 'NaN'
                elif np.isinf(value):
                    new_row[index] = 'Inf'
                else:
                    new_row[index] = np.asscalar(value)
            elif isinstance(value, np.ndarray) and value.ndim == 0:
                new_row[index] = value.item()
            elif hasattr(value, '__len__') and len(value) > 0:
                new_row[index] = str(value)

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
            # cells_to_update = cells_to_update[0] + [cells_to_update[-i] for i in range(len(cells_to_update)-1)]
        except:  # noqa
            sheet.append_row(new_row)

        # Run all necessary update operations
        if len(cells_to_update) > 0:
            sheet.update_cells(cells_to_update)
