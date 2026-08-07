import pandas as pd
import numpy as np
import sys
sys.path.append(r'C:/data_handler')

from muf_load_to_db import get_muf_schemas
from muf_load_to_db import get_muf_tables_names
from muf_load_to_db import get_time_muf_from_interpolated_muf
from muf_load_to_db import write_muf_to_db


def main():

    print('start')
    year=2023
    schemas = get_muf_schemas(year)
    # temp_df = pd.DataFrame(columns=[time])
    # print(schemas)
    for schema in schemas:
        print('month = ', schema[0])
        tables_names = get_muf_tables_names(schema[0], year)
        # print(tables_names)
        for table_name in tables_names:
            # print(table_name)
            split = table_name[0].split('_')
            # if len(split) > 1 and split[1] == 'interp':
            if len(split) == 1:
                print('day = ', table_name[0])
                muf = get_time_muf_from_interpolated_muf(table_name[0], schema[0], year)
                muf = pd.DataFrame(muf)
                muf.to_csv(str(schema[0].split('_')[1])+'_'+str(table_name[0].split('_')[0])+'.csv', header=False)
                # for val in muf:
                    # write_muf_to_db(table_name[0], schema[0], val[0], val[1], year)

                print('----- Schema range: '+str(table_name)+' - done -----')
    print('----- Done! -----')

main()