import psycopg2
from local_config import config

# host     = '10.10.65.82'
# database = 'recount_db_test'
# user     = 'postgres'
# password = '12345'
# port     = '1333'

class DbRequests:

    def __init__(self, filename, section = None,  reqwest= None, tuples= None) -> None:
        self.section = section
        self.filename = filename
        self.reqwest = reqwest
        self.tuples = tuples

        self.month_search = '%month%'

    def insert_many(self, reqwest, tuples, section):

        conn = None

        try:
            params = config(filename=self.filename, section=section)
            conn = psycopg2.connect(**params) 
            insert_data_reqest = conn.cursor() 

            insert_data_reqest.executemany(reqwest, tuples)
            insert_data_reqest.close()
            conn.commit()
            print('inserted')

        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
            
    def fetch_all(self, reqwest, section):

        conn = None
        
        try:
            # print('Connecting to the PostgreSQL database...')
            params = config(filename=self.filename, section=section)
            conn = psycopg2.connect(**params)
            select_reqest = conn.cursor()

            select_reqest.execute(reqwest)
            select_reqest_rows = select_reqest.fetchall()
            select_reqest.close()
            conn.commit()
                    
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
        
        return select_reqest_rows
        
    def execute_one_or_many(self, section, *reqwests):
        
        conn = None
        
        try:
            for reqwest in reqwests:
                print(reqwest)
                params = config(filename=self.filename, section=section)
                conn = psycopg2.connect(**params)
                select_reqest = conn.cursor()

                select_reqest.execute(reqwest)
                select_reqest.close()
                conn.commit()
            
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)

    def get_muf_schemas(self, year):

        conn = None
        section = 'muf_db_'+str(year)
        reqwest = f'SELECT nspname \
                    FROM pg_catalog.pg_namespace \
                    WHERE nspname LIKE \'{self.month_search}\''

        return self.fetch_all(reqwest, section)

    def get_muf_interp_tables_names(self, schema, year, station = 'cyprus1'):

        section = 'muf_db_'+str(year)
        reqwest = 'SELECT tablename as table FROM pg_catalog.pg_tables \
            WHERE schemaname = \'{0}\' and tablename ILIKE \'{1}_interp%\' '.format(str(schema), station)
        return self.fetch_all(reqwest, section)
    
    def get_muf_table_names(self, schema, year, station = 'cyprus1'):

        section = 'muf_db_'+str(year)
        reqwest = 'SELECT tablename as table FROM pg_catalog.pg_tables \
            WHERE schemaname = \'{0}\' and tablename NOT LIKE \'{1}\' ORDER BY tablename'.format(str(schema), '%interp%')
        return self.fetch_all(reqwest, section)

    def get_muf_table_content(self, table_name, schema, year):

        section = 'muf_db_'+str(year)
        reqwest = 'SELECT time, muf FROM "%s"."%s"' % (str(schema), str(table_name))

        return self.fetch_all(reqwest, section)
    
    def write_interpolated_muf_to_db(self, table_name, schema, df, year, transmit_station = 'cyprus1'):

        df = df.to_frame()
        df = df.reset_index().rename(columns={"index":"datetime"})	

        section = 'muf_db_'+str(year)

        print('loading interpolated muf data to database...')

        reqwest = 'CREATE TABLE IF NOT EXISTS "%s"."%s_interp_muf_day_%s" (datetime TIMESTAMP PRIMARY KEY, muf numeric NULL DEFAULT NULL)' % (schema, transmit_station, table_name)
        
        self.execute_one_or_many(section, reqwest)

        tuples = [tuple(x) for x in df.to_numpy()]
        cols = ','.join(list(df.columns))
        reqwest = 'INSERT INTO "%s"."%s_interp_muf_day_%s"(%s) VALUES (%%s, %%s)' % (schema, transmit_station, table_name, cols)
        print(reqwest)

        self.insert_many(reqwest, tuples, section)

        print('done loading interpolated muf data to database.')
        print('-----------------------------------------------')


def Load_muf_data_to_muf_db(year, month, day, time, muf, vrange,
            recive_station, transmit_station, muf_column, muf_row,
            filename='database2.ini'):

    conn = None
    section = 'muf_db_'+str(year)
    time_ = time

    try:
        # database="muf_db_"+year
        # params = config(database)
        print('Connecting to the PostgreSQL database...')
        params = config(filename=filename, section=section)
        conn = psycopg2.connect(**params)
        create_schema_reqest = conn.cursor()

        create_schema_reqest.execute('CREATE SCHEMA IF NOT EXISTS "month_%s"' % (str(month)))
        # print('CREATE SCHEMA IF NOT EXISTS %s' % (str(day_iri)))
        create_schema_reqest.execute('CREATE TABLE IF NOT EXISTS "month_%s"."%s" (time TIME NOT NULL, muf numeric NULL DEFAULT NULL, vrange numeric NULL DEFAULT NULL, recive_station VARCHAR(20), transmit_station VARCHAR(20), muf_column integer NULL DEFAULT NULL, muf_row integer NULL DEFAULT NULL)' % (str(month), str(day)))
        # print('CREATE TABLE IF NOT EXISTS %s."%s" (lat numeric NOT NULL, lon numeric NULL DEFAULT NULL, tec numeric NULL DEFAULT NULL, station VARCHAR(4))' % (str(day_iri), str(time_iri)))
        create_schema_reqest.close()
        conn.commit()
        # print('Database connection closed.')

        if muf > 0:

            conn = psycopg2.connect(**params)
            check_data_reqest = conn.cursor()
            reqest = 'select exists(select 1 from "month_%s"."%s" where muf_row = %s and muf_column = %s)' % (str(month), str(day), (muf_row), (muf_column))
            print(reqest)
            check_data_reqest.execute(reqest)
            # print('SELECT = ',reqest)
            check_data_reqest_row = check_data_reqest.fetchone()
            # print(check_data_reqest_row[0])

            if check_data_reqest_row[0] == False:

                insert_data_reqest = conn.cursor()
                reqest = 'INSERT INTO "month_%s"."%s" (time, muf, vrange, recive_station, transmit_station, muf_column, muf_row) VALUES (\'%s\', %s, %s, \'%s\', \'%s\', %s, %s)' % (str(month), str(day), time_, muf, vrange, recive_station, transmit_station, muf_column, muf_row)
                print(reqest)
                insert_data_reqest.execute(reqest)
                insert_data_reqest.close()
                conn.commit()

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
        pass

def write_interpolated_muf_to_db(tablename, schema, transmit_station, time, muf, interpolated):

    conn = None

    try:

        # print('Connecting to the PostgreSQL database...')
        params = config()
        conn = psycopg2.connect(**params)

        create_schema_reqest = conn.cursor()
        create_schema_reqest.execute('CREATE TABLE IF NOT EXISTS "%s"."%s_interp_muf_day_%s" (time TIME NOT NULL, muf numeric NULL DEFAULT NULL, interpolated boolean NULL DEFAULT NULL)' % (schema, transmit_station, tablename))
        create_schema_reqest.close()


        insert_data_reqest = conn.cursor()
        reqest = 'INSERT INTO "%s"."%s_interp_muf_day_%s" (time, muf, interpolated) VALUES (\'%s\', %s, %s)' % (schema, transmit_station, tablename, time, muf, interpolated)
        insert_data_reqest.execute(reqest)
        insert_data_reqest.close()
        conn.commit()

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
        pass

    # return select_reqest_rows

def write_minimuf_sunspots_to_db(tablename, schema, transmit_station, sunspot, threshold,  muf_f, time):

    conn = None

    try:

        # print('Connecting to the PostgreSQL database...')
        params = config()
        conn = psycopg2.connect(**params)

        create_schema_reqest = conn.cursor()
        create_schema_reqest.execute('CREATE TABLE IF NOT EXISTS "%s"."%s_minimuf_sunspots_day_%s" (time TIME NOT NULL, sunspot numeric NULL DEFAULT NULL, threshold numeric NULL DEFAULT NULL, muf_f numeric NULL DEFAULT NULL)' % (schema, transmit_station, tablename))
        create_schema_reqest.close()


        insert_data_reqest = conn.cursor()
        reqest = 'INSERT INTO "%s"."%s_minimuf_sunspots_day_%s" (time, sunspot, threshold, muf_f) VALUES (\'%s\', %s, %s, %s)' % (schema, transmit_station, tablename, time, sunspot, threshold, muf_f)
        insert_data_reqest.execute(reqest)
        insert_data_reqest.close()
        conn.commit()

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
        pass

def write_poly1d_degree_to_db(tablename, schema, transmit_station, degree, threshold,  muf_f, time):

    conn = None

    try:

        # print('Connecting to the PostgreSQL database...')
        params = config()
        conn = psycopg2.connect(**params)

        create_schema_reqest = conn.cursor()
        create_schema_reqest.execute('CREATE TABLE IF NOT EXISTS "%s"."%s_poly1d_day_%s" (time TIME NOT NULL, degree numeric NULL DEFAULT NULL, threshold numeric NULL DEFAULT NULL, muf_f numeric NULL DEFAULT NULL)' % (schema, transmit_station, tablename))
        create_schema_reqest.close()


        insert_data_reqest = conn.cursor()
        reqest = 'INSERT INTO "%s"."%s_poly1d_day_%s" (time, degree, threshold, muf_f) VALUES (\'%s\', %s, %s, %s)' % (schema, transmit_station, tablename, time, degree, threshold, muf_f)
        print(reqest)
        insert_data_reqest.execute(reqest)
        insert_data_reqest.close()
        conn.commit()

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
        pass

def check_db_minimuf_sunspots_table_exist(table_name, schema, transmit_station):

    conn = None

    try:

        params = config()
        conn = psycopg2.connect(**params)
        select_reqest = conn.cursor()
        reqest = 'SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = \'%s\' AND table_name = \'%s_minimuf_sunspots_day_%s\');' % (schema, transmit_station, table_name)
        # print(reqest)
        select_reqest.execute(reqest)
        select_reqest_rows = select_reqest.fetchone()
        select_reqest.close()
        conn.commit()

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
        pass

    return select_reqest_rows[0]

def check_db_poly1d_degree_table_exist(table_name, schema, transmit_station):

    conn = None

    try:

        params = config()
        conn = psycopg2.connect(**params)
        select_reqest = conn.cursor()
        reqest = 'SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = \'%s\' AND table_name = \'%s_poly1d_day_%s\');' % (schema, transmit_station, table_name)
        # print(reqest)
        select_reqest.execute(reqest)
        select_reqest_rows = select_reqest.fetchone()
        select_reqest.close()
        conn.commit()

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
        pass

    return select_reqest_rows[0]

def check_db_muf_interpolated_table_exist(table_name, schema, transmit_station):

    conn = None

    try:

        params = config()
        conn = psycopg2.connect(**params)
        select_reqest = conn.cursor()
        reqest = 'SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = \'%s\' AND table_name = \'%s_interp_muf_day_%s\');' % (schema, transmit_station, table_name)
        select_reqest.execute(reqest)
        select_reqest_rows = select_reqest.fetchone()
        select_reqest.close()
        conn.commit()

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
        pass

    return select_reqest_rows[0]

def get_muf_tables(table_name, schema, transmit_station, year,  filename='database.ini'):

    conn = None
    section = 'muf_db_'+str(year)

    try:

        # print('Connecting to the PostgreSQL database...')
        params = config(filename=filename, section=section)
        conn = psycopg2.connect(**params)
        select_reqest = conn.cursor()

        select_reqest.execute('SELECT time, muf FROM "%s"."%s" WHERE transmit_station = \'%s\'' % (str(schema), str(table_name), str(transmit_station)))
        select_reqest_rows = select_reqest.fetchall()
        select_reqest.close()
        conn.commit()

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
        pass

    return select_reqest_rows

def get_muf_transmit_station(table_name, schema):

    conn = None

    try:

        # print('Connecting to the PostgreSQL database...')
        params = config()
        conn = psycopg2.connect(**params)
        select_reqest = conn.cursor()

        select_reqest.execute('SELECT DISTINCT transmit_station FROM "%s"."%s"' % (str(schema), str(table_name)))
        select_reqest_rows = select_reqest.fetchall()
        select_reqest.close()
        conn.commit()

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
        pass

    return select_reqest_rows

def get_muf_tables_names(schema, year,  filename='database.ini'):

    conn = None
    section = 'muf_db_'+str(year)

    try:

        # print('Connecting to the PostgreSQL database...')
        params = config(filename=filename, section=section)
        conn = psycopg2.connect(**params)
        select_reqest = conn.cursor()

        reqwest = 'SELECT tablename as table FROM pg_catalog.pg_tables \
            WHERE schemaname = \'{0}\' and tablename ILIKE \'cyprus1_interp%\' '.format(str(schema))

        # print(reqest)

        select_reqest.execute(reqwest)
        select_reqest_rows = select_reqest.fetchall()
        select_reqest.close()
        conn.commit()

        select_reqest_rows.sort()

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
        pass

    return select_reqest_rows

def get_origin_muf_tables_names(schema, year, filename='database.ini'):

    conn = None
    section = 'muf_db_'+str(year)

    try:

        # print('Connecting to the PostgreSQL database...')
        params = config(filename=filename, section=section)
        conn = psycopg2.connect(**params)
        select_reqest = conn.cursor()

        reqwest = 'SELECT tablename as table FROM pg_catalog.pg_tables \
            WHERE schemaname = \'{0}\' '.format(str(schema))

        # print(reqest)

        select_reqest.execute(reqwest)
        select_reqest_rows = select_reqest.fetchall()
        select_reqest.close()
        conn.commit()

        select_reqest_rows.sort()

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
        pass

    return select_reqest_rows

def get_muf_schemas(year, filename='database.ini'):

    conn = None
    section = 'muf_db_'+str(year)

    try:

        # print('Connecting to the PostgreSQL database...')
        params = config(filename=filename, section=section)
        conn = psycopg2.connect(**params)
        select_reqest = conn.cursor()
        print('Connected')

        select_reqest.execute('SELECT nspname FROM pg_catalog.pg_namespace;')
        select_reqest_rows = select_reqest.fetchall()
        select_reqest.close()
        conn.commit()

        schema = []
        for row in select_reqest_rows:
            split_row = row[0].split('_')

            if split_row[0] == 'month':
                schema.append(row)

        return schema

    except (Exception, psycopg2.DatabaseError) as error:
        print('Error')
        print(error)
        # pass

def get_muf_data(table_name, schema, year, filename='database.ini'):

    conn = None
    section = 'muf_db_'+str(year)

    reqwest = 'SELECT time, muf FROM "%s"."%s"' % (str(schema), str(table_name))
    muf_data = Fetch_all(section, reqwest)
    
    return muf_data

def get_muf_schemas_Absol():

    conn = None

    try:

        # print('Connecting to the PostgreSQL database...')
        params = config()
        # print(params)
        conn = psycopg2.connect(**params)
        select_reqest = conn.cursor()

        select_reqest.execute('SELECT nspname FROM pg_catalog.pg_namespace;')
        select_reqest_rows = select_reqest.fetchall()
        select_reqest.close()
        conn.commit()

        schema = []
        for row in select_reqest_rows:
            split_row = row[0].split('_')


            if split_row[0] == 'day':
                schema.append(row)

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
        pass

    return schema

def get_time_muf_from_interpolated_muf(table_name, schema):


    conn = None

    try:

        # print('Connecting to the PostgreSQL database...')
        params = config()
        conn = psycopg2.connect(**params)
        select_reqest = conn.cursor()

        select_reqest.execute('SELECT time, muf FROM "%s"."%s"' % (str(schema), str(table_name)))
        select_reqest_rows = select_reqest.fetchall()
        select_reqest.close()
        conn.commit()

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
        pass

    return select_reqest_rows

def get_time_sunspots_from_interpolated_muf(table_name, schema):


    conn = None

    try:

        # print('Connecting to the PostgreSQL database...')
        params = config()
        conn = psycopg2.connect(**params)
        select_reqest = conn.cursor()

        select_reqest.execute('SELECT time, sunspot, muf_f FROM "%s"."%s"' % (str(schema), str(table_name)))
        select_reqest_rows = select_reqest.fetchall()
        select_reqest.close()
        conn.commit()

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
        pass

    return select_reqest_rows

def get_time_degree_from_interpolated_muf(table_name, schema):


    conn = None

    try:

        # print('Connecting to the PostgreSQL database...')
        params = config()
        conn = psycopg2.connect(**params)
        select_reqest = conn.cursor()

        select_reqest.execute('SELECT time, degree, muf_f FROM "%s"."%s"' % (str(schema), str(table_name)))
        select_reqest_rows = select_reqest.fetchall()
        select_reqest.close()
        conn.commit()

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
        pass

    return select_reqest_rows

def get_muf_transmit_station(table_name, schema, year):

    conn = None
    section = 'muf_db_'+str(year)

    try:

        # print('Connecting to the PostgreSQL database...')
        params = config(filename='database.ini',section=section)
        conn = psycopg2.connect(**params)
        select_reqest = conn.cursor()

        select_reqest.execute('SELECT DISTINCT transmit_station FROM "%s"."%s"' % (str(schema), str(table_name)))
        select_reqest_rows = select_reqest.fetchall()
        select_reqest.close()
        conn.commit()

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
        pass

    return select_reqest_rows

def get_abs_tec_data(table_name, schema):

    conn = None

    try:

        # print('Connecting to the PostgreSQL database...')
        params = config()
        conn = psycopg2.connect(**params)
        # print(conn)
        select_reqest = conn.cursor()

        select_reqest.execute('SELECT ut, i_v FROM "%s"."%s"' % (str(schema), str(table_name)))
        select_reqest_rows = select_reqest.fetchall()
        select_reqest.close()
        conn.commit()

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
        pass

    return select_reqest_rows

# ----------------------------------------------------------------------------

# def get_muf_data(table, schema, year):

#     section = 'muf_db_'+str(year)

#     reqwest = 'SELECT EXISTS ( SELECT FROM information_schema.tables WHERE  \
#         table_schema = \'%s\' AND table_name = \'%s\');' % (str(schema), str(table))

#     table_extst = Fetch_all(section, reqwest)


#     if table_extst[0][0]:
        
#         reqwest = 'SELECT time, muf FROM "%s"."%s"' % (str(schema), str(table))
#         tec_data = Fetch_all(section, reqwest)
        
#         return tec_data
        
#     else:
#         print('schema or table is not exist ' + schema +'.'+ table)
#         return None

def Insert_many(section, filename, reqwest, tuples):

    conn = None

    try:
        params = config(filename=filename, section=section)
        conn = psycopg2.connect(**params) 
        insert_data_reqest = conn.cursor() 

        insert_data_reqest.executemany(reqwest, tuples)
        insert_data_reqest.close()
        conn.commit()
        print('inserted')


    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
        
def Fetch_all(section, reqwest, filename='database.ini'):

    conn = None
    
    try:
        # print('Connecting to the PostgreSQL database...')
        params = config(filename=filename, section=section)
        conn = psycopg2.connect(**params)
        select_reqest = conn.cursor()
        select_reqest.execute(reqwest)
        select_reqest_rows = select_reqest.fetchall()
        select_reqest.close()
        conn.commit()
                
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    
    return select_reqest_rows
    
def Execute_many(section=None, filename='database.ini', *reqwests):
    
    conn = None
    
    try:
        for reqwest in reqwests:

            params = config(filename=filename, section=section)

            conn = psycopg2.connect(**params)
            select_reqest = conn.cursor()
            select_reqest.execute(reqwest)
            select_reqest.close()
            conn.commit()
        
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)