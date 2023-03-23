#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
db = "~/tts_data/forced_alignment"

with open (os.path.join(db,'hanna_mod_id4051_itr4_sig85_nf1k.csv')) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=';')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            print(f'\t{row[0]} works in the {row[1]} department, and was born in {row[2]}.')
            line_count += 1
    print(f'Processed {line_count} lines.')
